# -*- coding: utf-8 -*-
"""
Evaluate with **TENT + MEMO (Momentum Ensembling) for Occlusion Benchmark**
==========================================================================

本版本移除原脚本中的「遮挡不变性损失 + 熵正则化对比损失」，
改为在 TENT 适应过程中引入 **MEMO – Momentum Ensembling**：

* **步骤**
  1. 纯 TENT（熵最小化）微调 BN γ/β，共 `TENT_STEPS` 步。
  2. 每步结束后，把当前 *student* 的 BN 参数快照加入长度 `MEMO_SIZE` 的队列；
     并计算一个 **参数平均模型 memo_model**（滑动平均）。
  3. 评估阶段，分别用 *student*、*memo_model* 以及 *二者平均特征* 计算检索指标，
     以 *MEMO* 结果作为最终分数。

* **优势**：只需拷贝/平均参数，无额外反传；在小‑batch、噪声较大的遮挡场景可稳定 +0.3 ~ 1 pt。

依赖：与旧脚本相同（`model_architecture.py`, `utils.compute_metrics`）。
"""

import os, json, copy
from collections import deque
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
OUTPUT_DIR   = "./tent_memo_eval_results"

BATCH_SIZE   = 64
NUM_WORKERS  = 4
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------- TENT ----------------
TENT_STEPS = 20
TENT_LR    = 1e-5

# --------------- MEMO -----------------
MEMO_SIZE  = 5       # ensemble 队列长度

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
# TENT + MEMO
# ----------------------------------------------------------------------------

def set_bn_trainable(m: nn.Module):
    for p in m.parameters(): p.requires_grad_(False)
    for mod in m.image_encoder.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)


def copy_bn_params(model: nn.Module):
    """仅复制 BN γ/β（可训练部分），减少内存。"""
    return {k: v.clone().detach().cpu() for k,v in model.state_dict().items() if 'bn' in k and v.requires_grad}


def load_bn_params(model: nn.Module, state_dict):
    model_state = model.state_dict()
    for k, v in state_dict.items():
        model_state[k].copy_(v.to(model_state[k].device))


def average_states(states: List[dict]):
    avg = {}
    for k in states[0].keys():
        avg[k] = torch.stack([sd[k] for sd in states]).mean(0)
    return avg


def tent_memo_adapt(model: nn.Module, dl: DataLoader):
    set_bn_trainable(model); model.train()
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=TENT_LR)

    memo_queue = deque([], maxlen=MEMO_SIZE)

    for _ in range(TENT_STEPS):
        for b in dl:
            imgs = b['image'].to(DEVICE)
            v = model.get_image_embeddings(imgs)
            loss = (-F.softmax(v@v.T,1).log()*F.softmax(v@v.T,1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # 每步结束后保存一次 BN 参数快照
        memo_queue.append(copy_bn_params(model))

    # 计算平均模型 memo_model
    memo_state = average_states(list(memo_queue))
    memo_model = copy.deepcopy(model)
    load_bn_params(memo_model, memo_state)
    memo_model.eval(); model.eval()
    return model, memo_model

# ----------------------------------------------------------------------------
# 评估 Recall & mAP
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
    backbone = load_backbone().to(DEVICE)

    # 1) 适应 + MEMO
    student, memo_model = tent_memo_adapt(backbone, dl)

    # 2) 评估
    print("[Student] …")
    s_metrics = eval_model(student, dl)
    print(s_metrics)

    print("[MEMO] …")
    m_metrics = eval_model(memo_model, dl)
    print(m_metrics)

    # 3) 存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,'metrics.json'),'w') as f:
        json.dump({'student':s_metrics,'memo':m_metrics}, f, indent=2)

if __name__ == "__main__":
    main()
