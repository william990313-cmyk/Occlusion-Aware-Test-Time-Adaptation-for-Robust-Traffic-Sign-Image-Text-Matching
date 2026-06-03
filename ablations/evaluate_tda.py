#!/usr/bin/env python3
# ------------------------------------------------------------
#  TENT  +  TDA (Training-free Dynamic Adapter) 评估脚本
# ------------------------------------------------------------
#  • Stage-1: 纯 TENT —— 熵最小化，仅更新 BN γ/β
#  • Stage-2: TDA   —— 运行时 Key-Value 缓存注入（零梯度）
# ------------------------------------------------------------
import os, json, torch, math, torch.nn.functional as F
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm

# ========== 路径（按自己目录修改） ==========
MODEL_PATH = "checkpoints/model_best.pth.tar"
DATA_DIR   = "data/val_images"
META_PATH  = "data/val_metadata.json"
OUTPUT_DIR = "./tent_tda_eval_results"
# ===========================================

DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS= 4

# ---- TENT ----
TENT_STEPS = 10
TENT_LR    = 1e-5

# ---- TDA 参数 ----
CACHE_SIZE = 512
CONF_TH    = 0.70   # 写入缓存的余弦阈值
ALPHA_IMG  = 0.20   # 注入权重
ALPHA_TXT  = 0.20

# =========================================================
# 1) 数据集
# =========================================================
class ValDS(Dataset):
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
        it  = self.meta[idx]
        img = self.tf(Image.open(os.path.join(
                 self.root, os.path.basename(it["image_path"]))).convert("RGB"))
        cap = it.get("caption", it.get("captions", [""])[0])
        tok = self.tok(cap, padding='max_length',
                       truncation=True, max_length=128, return_tensors='pt')
        return {"image": img,
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0)}

def loader():
    return DataLoader(ValDS(DATA_DIR, META_PATH),
                      batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)

# =========================================================
# 2) 模型
# =========================================================
from model_architecture import create_model      # ← 若名字不同请改

def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    args  = ckpt.get("args", {})
    m = create_model(embed_dim=args.get("embed_dim",512),
                     image_model_type=args.get("image_model","resnet50"),
                     text_model_type=args.get("text_model","distilbert"))
    m.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return m

# =========================================================
# 3) TENT
# =========================================================
def freeze_except_bn(m):
    for p in m.parameters(): p.requires_grad_(False)
    for mod in m.image_encoder.modules():
        if isinstance(mod,(nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)

def entropy_loss(v,t):
    p = (v @ t.T).softmax(1)
    return (-p*(p.log()+1e-12)).sum(1).mean()

def tent_adapt(model, dl):
    if TENT_STEPS==0: return model.eval()
    freeze_except_bn(model); model.train()
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                     lr=TENT_LR)
    for _ in range(TENT_STEPS):
        for b in dl:
            v = model.get_image_embeddings(b["image"].to(DEVICE))
            t = model.get_text_embeddings(b["input_ids"].to(DEVICE),
                                          b["attention_mask"].to(DEVICE))
            loss = entropy_loss(v,t)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

# =========================================================
# 4) TDA 适配器
# =========================================================
def l2norm(x): return F.normalize(x, dim=-1)

class DynamicAdapter:
    def __init__(self, maxlen=CACHE_SIZE):
        self.cache_img = deque(maxlen=maxlen)
        self.cache_txt = deque(maxlen=maxlen)

    def mean_vecs(self):
        if not self.cache_img: return None, None
        return (torch.stack(list(self.cache_img)).mean(0),
                torch.stack(list(self.cache_txt)).mean(0))

    def maybe_update(self,img_emb,txt_emb):
        sim = (img_emb @ txt_emb.T).diag()
        msk = sim >= CONF_TH
        if msk.any():
            self.cache_img.extend(img_emb[msk].cpu())
            self.cache_txt.extend(txt_emb[msk].cpu())

# =========================================================
# 5) 评估（TDA 动态注入）
# =========================================================
@torch.no_grad()
def evaluate_tda(model, dl):
    # 自动推断嵌入维度，构建适配器
    first = next(iter(dl))
    dim = model.get_image_embeddings(first["image"].to(DEVICE)).size(1)
    adapter = DynamicAdapter(maxlen=CACHE_SIZE)

    img_all, txt_all = [], []
    for b in tqdm(dl, desc="Infer+TDA"):
        img_e = l2norm(model.get_image_embeddings(b["image"].to(DEVICE)))
        txt_e = l2norm(model.get_text_embeddings(b["input_ids"].to(DEVICE),
                                                 b["attention_mask"].to(DEVICE)))
        cache_img, cache_txt = adapter.mean_vecs()
        if cache_img is not None:
            img_e = l2norm(img_e + ALPHA_IMG*cache_img.to(img_e))
            txt_e = l2norm(txt_e + ALPHA_TXT*cache_txt.to(txt_e))
        adapter.maybe_update(img_e, txt_e)

        img_all.append(img_e.cpu()); txt_all.append(txt_e.cpu())

    v = torch.cat(img_all); t = torch.cat(txt_all)
    return compute_metrics(v,t)

def compute_metrics(v,t):
    sim  = v @ t.T
    rank = sim.argsort(1, descending=True)
    lab  = torch.arange(sim.size(0))
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dl    = loader()
    model = load_model().to(DEVICE)

    print(">> Stage-1  TENT 适应")
    model = tent_adapt(model, dl)

    print(">> Stage-2  TDA 动态注入 & 评估")
    res = evaluate_tda(model, dl); print(res)
    json.dump(res, open(os.path.join(OUTPUT_DIR,"metrics.json"),"w"), indent=2)

if __name__ == "__main__":
    main()
