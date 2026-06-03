# -*- coding: utf-8 -*-
"""
Evaluate & adapt a vision–language retrieval model on an occlusion benchmark
**with TENT + SAM + Selective BN**.

- **TENT**: entropy minimisation, only updates BN affine params.
- **SAM (Sharpness‑Aware Minimization)**: two–step update to seek flat minima.
- **Selective BN**: 仅开放高层 BatchNorm (如 backbone.layer4 & projection head) 参与适应，保持浅层稳定。

NOTE: 本脚本与 `evaluate_oata_tent_meanteacher_eata.py` 结构一致，
      仅替换适应策略；其余数据加载与评估接口保持相同，
      便于直接横向对比。
"""

import os, json, copy
from typing import List
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_PATH = "checkpoints/model_best.pth.tar"
DATA_DIR   = "data/val_images"
META_PATH  = "data/val_metadata.json"
OUTPUT_DIR = "./sam_selBN_eval_results"

BATCH_SIZE  = 64
NUM_WORKERS = 4
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------- TENT ----------------
TENT_STEPS = 5            # fewer steps because SAM double backward
TENT_LR    = 5e-6         # 半学习率，配合 SAM

# ---------------- SAM -----------------
SAM_RHO = 0.05            # perturbation radius（相对 γ/β 均值的百分比）

# -----------------------------------------------------------------------------
# Dataset (same as previous script, abbreviated)
# -----------------------------------------------------------------------------
class ITMDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, meta: str):
        self.root = root
        self.meta = json.load(open(meta))
        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        m = self.meta[idx]
        img = self.tf(Image.open(os.path.join(self.root, os.path.basename(m["image_path"]))).convert("RGB"))
        cap = m.get("caption", m.get("captions", [""])[0])
        tok = self.tok(cap, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            "image": img,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0)
        }

def loader():
    return DataLoader(ITMDataset(DATA_DIR, META_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -----------------------------------------------------------------------------
# Model utils
# -----------------------------------------------------------------------------
from model_architecture import create_model  # noqa

def load_model():
    ck = torch.load(MODEL_PATH, map_location=DEVICE)
    args = ck.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim",512),
                         image_model_type=args.get("image_model","resnet50"),
                         text_model_type=args.get("text_model","distilbert"))
    model.load_state_dict(ck.get("state_dict", ck))
    return model

# -----------------------------------------------------------------------------
# Selective BN utilities
# -----------------------------------------------------------------------------
SEL_BN_KEYWORDS = ["layer4", "proj_bn", "text_proj_bn"]

def set_selective_bn_trainable(model: nn.Module):
    for n,m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            train_flag = any(k in n for k in SEL_BN_KEYWORDS)
            m.weight.requires_grad_(train_flag)
            m.bias.requires_grad_(train_flag)
        else:
            for p in m.parameters(): p.requires_grad_(False)

# -----------------------------------------------------------------------------
# TENT + SAM adaptation
# -----------------------------------------------------------------------------

def entropy_loss(logits):
    p = F.softmax(logits, dim=1)
    return (-p * torch.log(p+1e-10)).sum(1).mean()


def sam_perturb(params: List[torch.Tensor], rho: float):
    grad_norm = torch.norm(torch.stack([p.grad.detach().flatten().norm() for p in params]))
    if grad_norm == 0: return [torch.zeros_like(p) for p in params]
    scale = rho / (grad_norm + 1e-12)
    e_ws = []
    with torch.no_grad():
        for p in params:
            e_w = p.grad * scale
            p.add_(e_w)  # ascend step
            e_ws.append(e_w)
    return e_ws


def sam_restore(params: List[torch.Tensor], e_ws):
    with torch.no_grad():
        for p,e in zip(params, e_ws):
            p.sub_(e)


def adapt(model: nn.Module, dl: DataLoader):
    set_selective_bn_trainable(model)
    model.train()
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=TENT_LR)

    for _ in range(TENT_STEPS):
        for batch in dl:
            imgs = batch["image"].to(DEVICE)
            ids  = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)

            # === first forward / backward ===
            v = model.get_image_embeddings(imgs)
            t = model.get_text_embeddings(ids, attn)
            logits = v @ t.T
            loss = entropy_loss(logits)
            opt.zero_grad(); loss.backward()

            bn_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            e_ws = sam_perturb(bn_params, SAM_RHO)

            # === second forward with perturbed weights ===
            v2 = model.get_image_embeddings(imgs)
            t2 = model.get_text_embeddings(ids, attn)
            loss2 = entropy_loss(v2 @ t2.T)
            opt.zero_grad(); loss2.backward(); opt.step()

            sam_restore(bn_params, e_ws)  # restore original weights post update
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Eval util (reuse)
# -----------------------------------------------------------------------------
from utils import compute_metrics  # noqa

def evaluate(model: nn.Module, dl: DataLoader):
    model.eval()
    img_emb, txt_emb = [], []
    with torch.no_grad():
        for b in tqdm(dl, desc="infer"):
            img_emb.append(model.get_image_embeddings(b["image"].to(DEVICE)).cpu())
            txt_emb.append(model.get_text_embeddings(b["input_ids"].to(DEVICE),
                                                    b["attention_mask"].to(DEVICE)).cpu())
    img_emb = torch.cat(img_emb); txt_emb = torch.cat(txt_emb)
    return compute_metrics(img_emb, txt_emb, list(range(len(img_emb))))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    dl = loader()
    model = load_model().to(DEVICE)

    print("[Before] baseline metrics …")
    print(evaluate(model, dl))

    model = adapt(model, dl)

    print("[After] SAM+SelBN metrics …")
    metrics = evaluate(model, dl)
    print(metrics)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
