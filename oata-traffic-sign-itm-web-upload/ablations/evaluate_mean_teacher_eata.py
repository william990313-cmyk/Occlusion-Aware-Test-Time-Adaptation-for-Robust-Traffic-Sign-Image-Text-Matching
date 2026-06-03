# -*- coding: utf-8 -*-
"""
Evaluate & adapt a vision–language retrieval model on an occlusion benchmark **with TENT + Mean‑Teacher + EATA**.

* Differences to the original script provided by the user
  1. **Occlusion‑invariant loss / entropy‑contrastive branch removed**;
  2. **Mean‑Teacher** — keep an exponential‑moving‑average (EMA) copy of the model;
  3. **EATA filtering** — only low‑entropy samples (< τ) contribute to the entropy loss and BN update.

Other command‑line arguments / dataset processing remain unchanged.

This script is self‑contained and can be dropped into the previous project folder.
"""

import os
import json
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T

# -----------------------------------------------------------------------------
# Config — edit here or override with argparse / environment variables
# -----------------------------------------------------------------------------
MODEL_PATH = "checkpoints/model_best.pth.tar"
DATA_DIR   = "data/val_images"
META_PATH  = "data/val_metadata.json"
OUTPUT_DIR = "./oata_eval_results"

BATCH_SIZE  = 64
NUM_WORKERS = 4
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------- TENT hyper‑params ----------------
TENT_STEPS = 10        # #BN adaptation iterations
TENT_LR    = 1e-5      # LR for BN γ/β parameters

# ---------------- Mean‑Teacher ----------------
EMA_MOMENTUM = 0.999    # teacher = m*teacher + (1-m)*student

# ---------------- EATA filtering --------------
EATA_TAU = 0.5          # keep samples with entropy < τ*log(C)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ITMOcclusionDataset(Dataset):
    """Minimal dataset wrapper for (image, caption) pairs with optional occlusion."""

    def __init__(self, data_dir: str, meta_path: str):
        self.data_dir = data_dir
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # basic transf.
        self.size = 224
        self.tf = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_len = 128

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        img_path = os.path.join(self.data_dir, os.path.basename(item["image_path"]))
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        caption = item.get("caption", item.get("captions", [""])[0])
        tok = self.tokenizer(
            caption, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "image": img,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0)
        }


def build_loader() -> DataLoader:
    ds = ITMOcclusionDataset(DATA_DIR, META_PATH)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -----------------------------------------------------------------------------
# Model helpers (assume create_model util exists)
# -----------------------------------------------------------------------------
from model_architecture import create_model  # noqa: E402


def load_pretrained() -> nn.Module:
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    args = ckpt.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim", 512),
                         image_model_type=args.get("image_model", "resnet50"),
                         text_model_type=args.get("text_model", "distilbert"))
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model

# -----------------------------------------------------------------------------
# TENT utilities (BN affine adaptation only)
# -----------------------------------------------------------------------------

def set_bn_trainable(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
        else:
            for p in m.parameters():
                p.requires_grad_(False)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return (-p * torch.log(p + 1e-10)).sum(1).mean()

# -----------------------------------------------------------------------------
# Mean‑Teacher helper
# -----------------------------------------------------------------------------

def update_ema(student: nn.Module, teacher: nn.Module, m: float = EMA_MOMENTUM):
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)

# -----------------------------------------------------------------------------
# Adaptation loop (TENT + Mean‑Teacher + EATA)
# -----------------------------------------------------------------------------

def adapt(model: nn.Module, loader: DataLoader):
    set_bn_trainable(model)
    model.train()

    # Create EMA teacher
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=TENT_LR)

    C = 1000  # dummy; will infer dynamically later
    for step in range(TENT_STEPS):
        for batch in loader:
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)

            # forward student
            img_emb_s = model.get_image_embeddings(images)
            txt_emb_s = model.get_text_embeddings(input_ids, attn)
            logits_s = torch.matmul(img_emb_s, txt_emb_s.t())  # similarity as logits
            C = logits_s.size(1)

            # entropy per sample
            ent = (-F.softmax(logits_s, dim=1).log() * F.softmax(logits_s, dim=1)).sum(1)
            thresh = EATA_TAU * np.log(C)
            mask = ent < thresh
            if mask.sum() == 0:
                continue  # skip batch if all high‑entropy

            loss = entropy_loss(logits_s[mask])
            opt.zero_grad(); loss.backward(); opt.step()

            # EMA update
            update_ema(model, teacher)

    model.eval()
    return model, teacher

# -----------------------------------------------------------------------------
# Evaluation (Recall@K, mAP) utils (re‑use compute_metrics)
# -----------------------------------------------------------------------------
from utils import compute_metrics  # noqa: E402


def evaluate(model: nn.Module, loader: DataLoader, use_teacher: bool = False, teacher: nn.Module = None):
    model.eval();
    if teacher is not None and use_teacher:
        eval_model = teacher
    else:
        eval_model = model

    img_embs: List[torch.Tensor] = []
    txt_embs: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            img_embs.append(eval_model.get_image_embeddings(images).cpu())
            txt_embs.append(eval_model.get_text_embeddings(input_ids, attn).cpu())

    img_embs = torch.cat(img_embs, 0)
    txt_embs = torch.cat(txt_embs, 0)
    metrics = compute_metrics(img_embs, txt_embs, list(range(len(img_embs))))
    return metrics

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    loader = build_loader()
    model = load_pretrained().to(DEVICE)

    print("[Before adaptation] evaluating …")
    base_metrics = evaluate(model, loader)
    print(base_metrics)

    print("[Adapt] TENT + Mean‑Teacher + EATA …")
    adapted_model, teacher = adapt(model, loader)

    print("[After adaptation] student metrics …")
    student_metrics = evaluate(adapted_model, loader)
    print(student_metrics)

    print("[After adaptation] teacher metrics …")
    teacher_metrics = evaluate(adapted_model, loader, use_teacher=True, teacher=teacher)
    print(teacher_metrics)

    # save simple json
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({"base": base_metrics, "student": student_metrics, "teacher": teacher_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
