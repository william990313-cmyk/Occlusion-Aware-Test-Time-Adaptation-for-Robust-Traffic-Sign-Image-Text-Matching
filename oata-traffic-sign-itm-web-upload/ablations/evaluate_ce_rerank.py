#!/usr/bin/env python3
# ------------------------------------------------------------
#  TENT  +  Cross-Encoder Re-Ranking 评估脚本
# ------------------------------------------------------------
#  1. 纯 TENT：BN γ/β 适应（无源数据）
#  2. Dual-Encoder 获取粗排相似度
#  3. 对每行 Top-K 用交叉编码器重排
#  4. 输出 Recall@1/5/10 与 mAP
# ------------------------------------------------------------
import os, json, torch, math, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)
from tqdm import tqdm

# ================= 路径常量（改成实际路径） =================
MODEL_PATH = "checkpoints/model_best.pth.tar"
DATA_DIR   = "data/val_images"
META_PATH  = "data/val_metadata.json"
OUTPUT_DIR = "./tent_ce_eval_results"
# ===========================================================

DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS= 4

# ---- TENT ----
TENT_STEPS = 10
TENT_LR    = 1e-5

# ---- Cross-Encoder Re-Rank ----
TOPK       = 64
CE_ID      = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # 公开模型

# ===========================================================
# 1) 数据集
# ===========================================================
class ValDataset(Dataset):
    def __init__(self, root, meta_json):
        self.meta = json.load(open(meta_json))
        self.root = root
        self.tf   = T.Compose([
            T.Resize((224,224), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        self.tok  = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        img_path = os.path.join(self.root,
                                os.path.basename(item["image_path"]))
        img = self.tf(Image.open(img_path).convert("RGB"))
        cap = item.get("caption", item.get("captions", [""])[0])

        tok = self.tok(cap, padding='max_length', truncation=True,
                       max_length=128, return_tensors='pt')
        return {
            "image": img,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "caption": cap                     # 直接返回原文字
        }

def build_loader():
    return DataLoader(ValDataset(DATA_DIR, META_PATH),
                      batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)

# ===========================================================
# 2) 双塔模型（保持你工程接口）
# ===========================================================
from model_architecture import create_model          # ← 如名称不同请改

def load_backbone():
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    args  = ckpt.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim",512),
                         image_model_type=args.get("image_model","resnet50"),
                         text_model_type=args.get("text_model","distilbert"))
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model

# ===========================================================
# 3) TENT 适应
# ===========================================================
def freeze_except_bn(m):
    for p in m.parameters(): p.requires_grad_(False)
    for mod in m.image_encoder.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)

def entropy_loss(v, t):
    p = (v @ t.T).softmax(1)
    return (-p * (p.log() + 1e-12)).sum(1).mean()

def tent_adapt(model, dl):
    if TENT_STEPS == 0:
        model.eval(); return model
    freeze_except_bn(model); model.train()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=TENT_LR)
    for _ in range(TENT_STEPS):
        for batch in dl:
            v = model.get_image_embeddings(batch["image"].to(DEVICE))
            t = model.get_text_embeddings(batch["input_ids"].to(DEVICE),
                                          batch["attention_mask"].to(DEVICE))
            loss = entropy_loss(v, t)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

# ===========================================================
# 4) 交叉编码器
# ===========================================================
ce_tok   = AutoTokenizer.from_pretrained(CE_ID)
ce_model = AutoModelForSequenceClassification.from_pretrained(CE_ID).to(DEVICE)
ce_model.eval()

@torch.no_grad()
def ce_score(text_pairs):
    tok = ce_tok(text_pairs[0], text_pairs[1],
                 padding=True, truncation=True,
                 max_length=128, return_tensors='pt').to(DEVICE)
    return ce_model(**tok).logits.squeeze(-1)

# ===========================================================
# 5) 指标 + Re-Rank
# ===========================================================
@torch.no_grad()
def compute_metrics(dual_img, dual_txt, captions):
    sim   = dual_img @ dual_txt.T
    N     = sim.size(0)
    rank0 = sim.argsort(1, descending=True)             # (N,N)

    new_rank = torch.empty_like(rank0)
    for i in tqdm(range(N), desc="Cross-Encoder"):
        top_idx = rank0[i, :TOPK]
        pair1   = [captions[i]] * TOPK
        pair2   = [captions[j] for j in top_idx.tolist()]
        scores  = ce_score((pair1, pair2)).cpu()
        _, rer  = scores.sort(descending=True)
        new_rank[i] = torch.cat([top_idx[rer], rank0[i, TOPK:]], 0)

    lab = torch.arange(N)
    r1  = (new_rank[:,:1]==lab[:,None]).float().mean().item()*100
    r5  = (new_rank[:,:5]==lab[:,None]).any(1).float().mean().item()*100
    r10 = (new_rank[:,:10]==lab[:,None]).any(1).float().mean().item()*100
    eq   = (new_rank == lab[:,None])
    prec = eq.float().cumsum(1) / (torch.arange(N) + 1)
    mAP  = (prec * eq.float()).sum(1).mean().item()*100
    return {"R@1": round(r1,2), "R@5": round(r5,2),
            "R@10": round(r10,2), "mAP": round(mAP,2)}

# ===========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loader = build_loader()
    model  = load_backbone().to(DEVICE)

    print(">>> Stage-1  TENT BN-adapt")
    model  = tent_adapt(model, loader)

    print(">>> Stage-2  Dual-Encoder forward")
    img_emb, txt_emb, caps = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="DualEmb"):
            img_emb.append(model.get_image_embeddings(b["image"].to(DEVICE)).cpu())
            txt_emb.append(model.get_text_embeddings(b["input_ids"].to(DEVICE),
                                                     b["attention_mask"].to(DEVICE)).cpu())
            caps.extend(b["caption"])                     # 直接取字符串
    img_emb = torch.cat(img_emb, 0)
    txt_emb = torch.cat(txt_emb, 0)

    print(">>> Stage-3  Cross-Encoder Re-Rank")
    res = compute_metrics(img_emb, txt_emb, caps)
    print(res)

    json.dump(res, open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
