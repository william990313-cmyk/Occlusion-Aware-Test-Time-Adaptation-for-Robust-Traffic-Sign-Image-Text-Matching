# -*- coding: utf-8 -*-
"""
TENT + EATA-Plus 适应 & 评估
--------------------------------------------------
• Stage-1:  TENT 熵最小化  (更新 BN γ/β)
• Stage-2:  置信度过滤 + 一致性对比 (EATA-Plus)
--------------------------------------------------
路径常量 MODEL_CKPT / IMG_DIR / META_JSON / OUT_DIR
已保留原值，不做任何额外修改。
"""
import os, json, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 已有路径（无需改动） ========= #
MODEL_CKPT = "checkpoints/model_best.pth.tar"
IMG_DIR    = "data/val_images"
META_JSON  = "data/val_metadata.json"
OUT_DIR    = "./eata_plus_result"

DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS= 4

# ---------- TENT ---------- #
TENT_STEPS = 20
TENT_LR    = 1e-5

# ---------- EATA-Plus ---------- #
TAU_EATA   = 0.5        # 熵阈值系数，保留熵 < τ·logC 的样本
LAMBDA_CON = 0.5        # 一致性对比权重
TEMP       = 0.07       # InfoNCE 温度

# =========================================================
# 1) 数据集
# =========================================================
class ValDataset(Dataset):
    def __init__(self, root, meta_json):
        self.meta = json.load(open(meta_json))
        self.root = root
        # 关键： ToTensor 在最前，后面再做 Tensor-级增强
        self.tf_img = T.Compose([
            T.Resize((224,224), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        self.tf_aug = T.RandomHorizontalFlip(p=0.5)   # Tensor-compatible
        self.tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        img_path = os.path.join(self.root,
                                os.path.basename(item["image_path"]))
        img = self.tf_img(Image.open(img_path).convert("RGB"))
        img_aug = self.tf_aug(img.clone())             # 一致性视图
        cap = item.get("caption", item.get("captions", [""])[0])
        tok = self.tok(cap, padding='max_length', truncation=True,
                       max_length=128, return_tensors='pt')
        return {
            "image": img,
            "image_aug": img_aug,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0)
        }

def build_loader():
    ds = ValDataset(IMG_DIR, META_JSON)
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=NUM_WORKERS)

# =========================================================
# 2) 模型
# =========================================================
from model_architecture import create_model       # ← 与工程保持一致

def load_backbone():
    ckpt = torch.load(MODEL_CKPT, map_location="cpu")
    args = ckpt.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim",512),
                         image_model_type=args.get("image_model","resnet50"),
                         text_model_type=args.get("text_model","distilbert"))
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model

# =========================================================
# 3) TENT 支持函数
# =========================================================
def freeze_except_bn(m):
    for p in m.parameters(): p.requires_grad_(False)
    for mod in m.image_encoder.modules():
        if isinstance(mod,(nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)

def entropy_loss(v, t):          # v,t 已归一化
    logits = v @ t.T
    p = logits.softmax(1)
    return (-p * (p.log()+1e-12)).sum(1).mean()

# =========================================================
# 4) EATA-Plus 适应
# =========================================================
def adapt_tent_eata(model, dl):
    freeze_except_bn(model); model.train()
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                     lr=TENT_LR)

    C = BATCH_SIZE          # 近似 batch 相似度矩阵宽度
    H_TH = TAU_EATA * math.log(C)

    for _ in range(TENT_STEPS):
        for batch in dl:
            img  = batch["image"].to(DEVICE)
            img2 = batch["image_aug"].to(DEVICE)
            ids  = batch["input_ids"].to(DEVICE)
            att  = batch["attention_mask"].to(DEVICE)

            v1 = model.get_image_embeddings(img)      # (B,d)
            t1 = model.get_text_embeddings(ids, att)  # (B,d)

            # ---------- 置信度过滤 ----------
            logits = v1 @ t1.T
            ent = (-logits.softmax(1) *
                   logits.log_softmax(1)).sum(1)      # (B,)
            mask = ent < H_TH
            if mask.sum() == 0:           # 无低熵样本，跳过
                continue

            # ---------- 一致性视图 ----------
            v2 = model.get_image_embeddings(img2)      # (B,d)

            #  InfoNCE 双向一致性
            sim1 = v1 @ v2.T / TEMP
            sim2 = v2 @ v1.T / TEMP
            labels = torch.arange(img.size(0), device=DEVICE)
            loss_con = 0.5 * (F.cross_entropy(sim1, labels) +
                              F.cross_entropy(sim2, labels))

            # ---------- 总 Loss ----------
            loss_tent = entropy_loss(v1[mask], t1[mask])
            loss = loss_tent + LAMBDA_CON * loss_con

            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); return model

# =========================================================
# 5) 评估
# =========================================================
@torch.no_grad()
def calc_metrics(model, dl):
    img_all, txt_all = [], []
    for b in tqdm(dl, desc="Eval"):
        img_all.append(model.get_image_embeddings(
                       b["image"].to(DEVICE)).cpu())
        txt_all.append(model.get_text_embeddings(
                       b["input_ids"].to(DEVICE),
                       b["attention_mask"].to(DEVICE)).cpu())
    v = torch.cat(img_all,0); t = torch.cat(txt_all,0)
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

# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dl     = build_loader()
    model  = load_backbone().to(DEVICE)

    print(">>> 适应 (TENT + EATA-Plus)")
    model  = adapt_tent_eata(model, dl)

    print(">>> 评估...")
    res = calc_metrics(model, dl); print(res)
    with open(os.path.join(OUT_DIR,"metrics.json"),"w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    import math, json
    main()
