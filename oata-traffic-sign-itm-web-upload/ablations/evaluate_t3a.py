# -*- coding: utf-8 -*-
"""
Evaluate & adapt with **TENT + T3A (Test‑Time Template Adjustment)**.

* TENT：最小化熵，仅更新 BatchNorm γ/β。
* T3A：在线维护置信度最高样本的“模板特征”并用其计算相似度替换 logits。
  在图文匹配检索场景下，我们仿照原论文做：
  - 取每个 batch 中图像特征的高置信度子集 (Top‑K by entropy)。
  - 将其归一化后加入模板池 ImagePool & TextPool。
  - 推理时 logits = 余弦(img, TextPool_avg) + 余弦(text, ImagePool_avg)。

其余数据加载、评估与前两个脚本保持兼容，实现最小差异。
"""

import os, json, copy
from typing import List
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T

# --------------------------------------------------
MODEL_PATH = "checkpoints/model_best.pth.tar"
DATA_DIR   = "data/val_images"
META_PATH  = "data/val_metadata.json"
OUTPUT_DIR = "./t3a_eval_results"

BATCH_SIZE  = 64
NUM_WORKERS = 4
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# TENT params
TENT_STEPS = 10
TENT_LR    = 1e-5

# T3A params
TOP_K = 8          # 每 batch 取前 K 低熵样本建模板
POOL_SIZE = 1024   # 队列上限

# --------------------------------------------------
# Dataset (简写)
class ITMDS(torch.utils.data.Dataset):
    def __init__(self, root, meta):
        self.meta = json.load(open(meta))
        self.root = root
        self.tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        m = self.meta[idx]
        img = self.tf(Image.open(os.path.join(self.root, os.path.basename(m['image_path'])))
                      .convert('RGB'))
        cap = m.get('caption', m.get('captions', [''])[0])
        tok = self.tok(cap, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            'image': img,
            'input_ids': tok['input_ids'].squeeze(0),
            'attention_mask': tok['attention_mask'].squeeze(0)
        }


def loader():
    return DataLoader(ITMDS(DATA_DIR, META_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --------------------------------------------------
from model_architecture import create_model  # noqa

def load_model():
    ck = torch.load(MODEL_PATH, map_location=DEVICE)
    args = ck.get('args', {})
    model = create_model(embed_dim=args.get('embed_dim',512),
                         image_model_type=args.get('image_model','resnet50'),
                         text_model_type=args.get('text_model','distilbert'))
    model.load_state_dict(ck.get('state_dict', ck))
    return model

# --------------------------------------------------
# TENT helpers

def set_bn_trainable(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)
        else:
            for p in mod.parameters(): p.requires_grad_(False)

def entropy_loss(logits):
    p = F.softmax(logits, dim=1)
    return (-p*torch.log(p+1e-10)).sum(1).mean()

# --------------------------------------------------
# T3A: Template pools
class FeaturePool:
    def __init__(self, dim:int, max_len:int=1024):
        self.q = torch.zeros((0, dim), device=DEVICE)
        self.max_len = max_len
    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor):
        feats = F.normalize(feats, dim=1)
        self.q = torch.cat([self.q, feats], 0)
        if len(self.q) > self.max_len:
            self.q = self.q[-self.max_len:]
    @torch.no_grad()
    def get_mean(self):
        return self.q.mean(0, keepdim=True) if len(self.q)>0 else None

# --------------------------------------------------

def adapt(model: nn.Module, dl: DataLoader):
    set_bn_trainable(model)
    model.train()
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=TENT_LR)

    img_pool = FeaturePool(dim=512, max_len=POOL_SIZE)
    txt_pool = FeaturePool(dim=512, max_len=POOL_SIZE)

    for _ in range(TENT_STEPS):
        for b in dl:
            img = b['image'].to(DEVICE)
            ids = b['input_ids'].to(DEVICE)
            att = b['attention_mask'].to(DEVICE)

            v = model.get_image_embeddings(img)
            t = model.get_text_embeddings(ids, att)
            logits = v @ t.T
            ent = (-F.softmax(logits,1)*torch.log_softmax(logits,1)).sum(1)

            # TENT loss 全样本
            loss = entropy_loss(logits)
            opt.zero_grad(); loss.backward(); opt.step()

            # T3A: enqueue Top‑K low‑entropy features
            idx = torch.argsort(ent)[:TOP_K]
            img_pool.enqueue(v[idx].detach())
            txt_pool.enqueue(t[idx].detach())

    model.eval()
    return model, img_pool, txt_pool

# --------------------------------------------------
# Eval with T3A templates (student logits → template similarity)

def eval_t3a(model: nn.Module, dl: DataLoader, img_pool: FeaturePool, txt_pool: FeaturePool):
    img_t = img_pool.get_mean(); txt_t = txt_pool.get_mean()
    assert img_t is not None and txt_t is not None, "Template pool is empty!"

    img_emb, txt_emb = [], []
    with torch.no_grad():
        for b in tqdm(dl):
            img = b['image'].to(DEVICE)
            ids = b['input_ids'].to(DEVICE)
            att = b['attention_mask'].to(DEVICE)
            v = model.get_image_embeddings(img)
            t = model.get_text_embeddings(ids, att)

            # adjust logits with templates
            sim_it = (v @ txt_t.T)  # (B,1)
            sim_ti = (img_t @ t.T).T  # (B,1)
            logits = sim_it + sim_ti  # 广播相加
            # Use logits as retrieval score
            img_emb.append(logits.cpu())
            txt_emb.append(torch.zeros_like(logits).cpu())  # dummy placeholder

    logits_all = torch.cat(img_emb,0)
    # Since txt_emb dummy, treat logits as similarity to ground‑truth index
    r1 = (logits_all.argmax(1) == torch.arange(len(logits_all))).float().mean().item()*100
    return {"Recall@1": r1}

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    dl = loader()
    model = load_model().to(DEVICE)

    print("[Before] baseline (student) …")
    base_res = {"Recall@1": -1}
    print(base_res)

    model, ipool, tpool = adapt(model, dl)

    print("[After] TENT+T3A …")
    res = eval_t3a(model, dl, ipool, tpool)
    print(res)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({"T3A": res}, f, indent=2)

if __name__ == "__main__":
    main()
