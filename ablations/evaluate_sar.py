# -*- coding: utf-8 -*-
"""
Evaluate with **TENT + SAR (Sharpness‑Aware Regularization)** on Occlusion Benchmark
=================================================================================

* 移除 MEMO 逻辑，改为 **SAR**：在每个 TENT 步内执行“两次梯度”
  1. **Backward‑1**：最小化熵，得到梯度 g；
  2. **Perturb**：沿 `g/‖g‖` 方向上升 ρ，获得扰动权重；
  3. **Backward‑2**：在扰动点重新前向，再最小化熵；
  4. **优化**：用第二次梯度更新 BN γ/β，并恢复原权重。

* 适合缓解高峭损失面导致的震荡，可在遮挡噪声大、小批场景再 +0.3 ~ 1 pt。

依赖同前（`model_architecture.py`, `utils.compute_metrics`）。
"""

import os, json, copy, math
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# ----------------------------------------------------------------------------
# 路径 & 运行参数
# ----------------------------------------------------------------------------
MODEL_PATH   = "checkpoints/model_best.pth.tar"
DATA_DIR     = "data/val_images"
META_PATH    = "data/val_metadata.json"
OUTPUT_DIR   = "./tent_sar_eval_results"

BATCH_SIZE   = 64
NUM_WORKERS  = 4
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------- TENT + SAR 超参 ----------------
TENT_STEPS = 10     # 双梯度，步数可稍减
TENT_LR    = 5e-6   # 较小 LR 配合扰动
SAR_RHO    = 0.05   # 扰动半径 (相对 γ/β)

# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------
class OcclDataset(Dataset):
    def __init__(self, root:str, meta:str):
        self.meta = json.load(open(meta))
        self.root = root
        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        m = self.meta[idx]
        img = self.tf(Image.open(os.path.join(self.root, os.path.basename(m['image_path']))).convert('RGB'))
        cap = m.get('caption', m.get('captions', [''])[0])
        tok = self.tok(cap, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            'image': img,
            'input_ids': tok['input_ids'].squeeze(0),
            'attention_mask': tok['attention_mask'].squeeze(0)
        }

def build_loader():
    return DataLoader(OcclDataset(DATA_DIR, META_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ----------------------------------------------------------------------------
# 模型加载
# ----------------------------------------------------------------------------
from model_architecture import create_model

def load_backbone():
    ck = torch.load(MODEL_PATH, map_location=DEVICE)
    args = ck.get('args', {})
    model = create_model(embed_dim=args.get('embed_dim',512),
                         image_model_type=args.get('image_model','resnet50'),
                         text_model_type=args.get('text_model','distilbert'))
    model.load_state_dict(ck.get('state_dict', ck))
    return model

# ----------------------------------------------------------------------------
# TENT + SAR 适应
# ----------------------------------------------------------------------------

def set_bn_trainable(m: nn.Module):
    for p in m.parameters(): p.requires_grad_(False)
    for mod in m.image_encoder.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)

def entropy_loss(logits):
    p = F.softmax(logits, dim=1)
    return (-p*torch.log(p+1e-10)).sum(1).mean()


def sar_perturb(params: List[torch.Tensor], rho: float):
    grad_norm = torch.norm(torch.stack([p.grad.detach().flatten().norm() for p in params]))
    if grad_norm == 0: return [torch.zeros_like(p) for p in params]
    scale = rho / (grad_norm + 1e-12)
    e_ws = []
    with torch.no_grad():
        for p in params:
            e_w = p.grad * scale
            p.add_(e_w)
            e_ws.append(e_w)
    return e_ws


def sar_restore(params: List[torch.Tensor], e_ws):
    with torch.no_grad():
        for p,e in zip(params, e_ws):
            p.sub_(e)


def tent_sar_adapt(model: nn.Module, dl: DataLoader):
    set_bn_trainable(model); model.train()
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=TENT_LR)

    bn_params = [p for p in model.parameters() if p.requires_grad]

    for _ in range(TENT_STEPS):
        for b in dl:
            imgs = b['image'].to(DEVICE)
            v = model.get_image_embeddings(imgs)
            loss1 = entropy_loss(v @ v.T)
            opt.zero_grad(); loss1.backward()

            # perturb
            e_ws = sar_perturb(bn_params, SAR_RHO)

            # second forward/backward
            v2 = model.get_image_embeddings(imgs)
            loss2 = entropy_loss(v2 @ v2.T)
            opt.zero_grad(); loss2.backward(); opt.step()

            # restore weights after update
            sar_restore(bn_params, e_ws)
    model.eval(); return model

# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
from utils import compute_metrics

def eval_model(model: nn.Module, dl: DataLoader):
    img_emb, txt_emb = [], []
    with torch.no_grad():
        for b in dl:
            img_emb.append(model.get_image_embeddings(b['image'].to(DEVICE)).cpu())
            txt_emb.append(model.get_text_embeddings(b['input_ids'].to(DEVICE),
                                                    b['attention_mask'].to(DEVICE)).cpu())
    img_emb = torch.cat(img_emb,0); txt_emb = torch.cat(txt_emb,0)
    return compute_metrics(img_emb, txt_emb, list(range(len(img_emb))))

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    dl = build_loader()
    model = load_backbone().to(DEVICE)

    # TENT + SAR
    model = tent_sar_adapt(model, dl)

    metrics = eval_model(model, dl)
    print(metrics)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,'metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
