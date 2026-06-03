#!/usr/bin/env python3
# -------------------------------------------------------------
#  Evaluation :  TENT  +  TCR  (Test-time Contrastive Retrieval)
# -------------------------------------------------------------
#  • Stage-1 : TENT   —— 熵最小化，仅更新 BN γ/β
#  • Stage-2 : TCR    —— 批内构造伪正/负做行级 InfoNCE
# -------------------------------------------------------------
import os, json, math, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm

# ===================== 路径常量（请先改为实路径） =====================
MODEL_PATH  = "checkpoints/model_best.pth.tar"
DATA_DIR    = "data/val_images"
META_PATH   = "data/val_metadata.json"
OUTPUT_DIR  = "./tent_tcr_eval_results"
# ===================================================================

DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS= 4

# ---- TENT ----
TENT_STEPS = 20
TENT_LR    = 1e-5

# ---- TCR ----
POS_RATIO   = 0.10       # 前 P% 视为正
LAMBDA_CONTR= 0.5        # 对比损失权重
TEMP        = 0.07       # InfoNCE 温度

# ==============================================================
# 1) 数据集
# ==============================================================
class ValDataset(Dataset):
    def __init__(self, root, meta):
        self.meta = json.load(open(meta))
        self.root = root
        self.tf   = T.Compose([
            T.Resize((224,224), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.tok  = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        img  = self.tf(Image.open(os.path.join(
                    self.root, os.path.basename(item["image_path"]))).convert("RGB"))
        cap  = item.get("caption", item.get("captions", [""])[0])
        tok  = self.tok(cap, padding='max_length', truncation=True,
                        max_length=128, return_tensors='pt')
        return {
            "image": img,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0)
        }

def build_loader():
    return DataLoader(ValDataset(DATA_DIR, META_PATH),
                      batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)

# ==============================================================
# 2) 模型加载（保持你工程接口）
# ==============================================================
from model_architecture import create_model    # ← 若名字不同请改

def load_backbone():
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    args  = ckpt.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim",512),
                         image_model_type=args.get("image_model","resnet50"),
                         text_model_type=args.get("text_model","distilbert"))
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model

# ==============================================================
# 3) TENT 支持函数
# ==============================================================
def freeze_except_bn(m):
    for p in m.parameters(): p.requires_grad_(False)
    for mod in m.image_encoder.modules():
        if isinstance(mod,(nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)

def entropy_loss(v, t):
    logits = v @ t.T
    p = logits.softmax(1)
    return (-p * (p.log() + 1e-12)).sum(1).mean()

# ==============================================================
# 4) TENT + TCR 适应
# ==============================================================
def adapt_tent_tcr(model, dl):
    freeze_except_bn(model); model.train()
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                     lr=TENT_LR)

    for _ in range(TENT_STEPS):
        for batch in dl:
            img = batch["image"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            att = batch["attention_mask"].to(DEVICE)

            v = model.get_image_embeddings(img)          # (B,d)
            t = model.get_text_embeddings(ids, att)      # (B,d)

            # ---------------- TENT 熵最小化 ----------------
            loss_tent = entropy_loss(v, t)

            # ---------------- TCR 行级 InfoNCE -------------
            B = v.size(0)
            sim_i2t = v @ t.T              # (B,B)
            sim_t2i = t @ v.T

            k_pos   = max(1, int(POS_RATIO * B))    # 正样本 per 行
            topk_idx = sim_i2t.topk(k_pos, 1).indices      # (B,k_pos)
            row_idx  = torch.arange(B, device=DEVICE).unsqueeze(1).expand_as(topk_idx)

            logits_i2t  = sim_i2t / TEMP
            logits_t2i  = sim_t2i / TEMP

            # image → text
            pos_logsum  = logits_i2t[row_idx, topk_idx].logsumexp(1)
            denom_log   = logits_i2t.logsumexp(1)
            loss_i2t    = -(pos_logsum - denom_log).mean()

            # text → image
            pos_logsum2 = logits_t2i[row_idx, topk_idx].logsumexp(1)
            denom_log2  = logits_t2i.logsumexp(1)
            loss_t2i    = -(pos_logsum2 - denom_log2).mean()

            loss_contrast = 0.5 * (loss_i2t + loss_t2i)

            # ---------------- 总损失 ----------------
            loss = loss_tent + LAMBDA_CONTR * loss_contrast
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); return model

# ==============================================================
# 5) 评估
# ==============================================================
@torch.no_grad()
def calc_metrics(m, dl):
    v_all, t_all = [], []
    for b in tqdm(dl, desc="Eval"):
        v_all.append(m.get_image_embeddings(b["image"].to(DEVICE)).cpu())
        t_all.append(m.get_text_embeddings(b["input_ids"].to(DEVICE),
                                           b["attention_mask"].to(DEVICE)).cpu())
    v = torch.cat(v_all,0); t = torch.cat(t_all,0)
    sim = v @ t.T
    rank= sim.argsort(1, descending=True)
    lab = torch.arange(sim.size(0))
    r1  = (rank[:,:1]==lab[:,None]).float().mean().item()*100
    r5  = (rank[:,:5]==lab[:,None]).any(1).float().mean().item()*100
    r10 = (rank[:,:10]==lab[:,None]).any(1).float().mean().item()*100
    eq  = (rank==lab[:,None])
    prec= eq.float().cumsum(1)/(torch.arange(sim.size(0))+1)
    mAP = (prec*eq.float()).sum(1).mean().item()*100
    return {"R@1":round(r1,2),"R@5":round(r5,2),
            "R@10":round(r10,2),"mAP":round(mAP,2)}

# ==============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dl     = build_loader()
    model  = load_backbone().to(DEVICE)

    print(">>> Adapting (TENT + TCR) ...")
    model  = adapt_tent_tcr(model, dl)

    print(">>> Evaluating ...")
    res = calc_metrics(model, dl);  print(res)
    json.dump(res, open(os.path.join(OUTPUT_DIR,"metrics.json"),"w"), indent=2)

if __name__ == "__main__":
    import json
    main()
