#!/usr/bin/env python3
# ------------------------------------------------------------
#  TENT  +  FETTA  (Frustratingly-Easy TTA for VLMs)
# ------------------------------------------------------------
#    • Stage-1 : 纯 TENT   ——  BN γ/β 适应
#    • Stage-2 : FETTA    ——  微调文本 Prompt Token 前 N 维
# ------------------------------------------------------------
import os, json, math, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm

# ================= 路径（请改成真实路径） =================
MODEL_PATH = "checkpoints/model_best.pth.tar"
DATA_DIR   = "data/val_images"
META_PATH  = "data/val_metadata.json"
OUTPUT_DIR = "./fetta_eval_results"
# =========================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 64
NUM_WORKERS  = 4

# ---- TENT ----
TENT_STEPS   = 10
TENT_LR      = 1e-5

# ---- FETTA ----
PROMPT_DIM   = 10       # 只更新 Prompt 向量前 N 维
FETTA_STEPS  = 10
LR_PROMPT    = 5e-3

# =========================================================
# 1) 数据
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
        item = self.meta[idx]
        img  = self.tf(Image.open(os.path.join(
                    self.root, os.path.basename(item["image_path"]))).convert("RGB"))
        cap  = item.get("caption", item.get("captions", [""])[0])
        tok  = self.tok(cap, padding='max_length', truncation=True,
                        max_length=128, return_tensors='pt')
        return {"image": img,
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0)}

def build_loader():
    return DataLoader(ValDS(DATA_DIR, META_PATH),
                      batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)

# =========================================================
# 2) 模型
# =========================================================
from model_architecture import create_model          # ← 若名字不同请调整

def load_backbone():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    args = ckpt.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim",512),
                         image_model_type=args.get("image_model","resnet50"),
                         text_model_type=args.get("text_model","distilbert"))
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model

# =========================================================
# 3) TENT 工具
# =========================================================
def freeze_except_bn(m):
    for p in m.parameters(): p.requires_grad_(False)
    for l in m.image_encoder.modules():
        if isinstance(l,(nn.BatchNorm2d, nn.BatchNorm1d)):
            l.weight.requires_grad_(True)
            l.bias.requires_grad_(True)

def entropy_loss(v,t):
    p = (v @ t.T).softmax(1)
    return (-p*(p.log()+1e-12)).sum(1).mean()

def tent_adapt(model, dl):
    if TENT_STEPS==0: return model.eval()
    freeze_except_bn(model); model.train()
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=TENT_LR)
    for _ in range(TENT_STEPS):
        for b in dl:
            v = model.get_image_embeddings(b["image"].to(DEVICE))
            t = model.get_text_embeddings(b["input_ids"].to(DEVICE),
                                          b["attention_mask"].to(DEVICE))
            loss = entropy_loss(v,t)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

# =========================================================
# 4) FETTA Prompt Token
# =========================================================
def build_prompt_token(dim):
    tok = nn.Parameter(torch.zeros(1,1,dim))
    nn.init.normal_(tok, mean=0, std=0.02)
    mask = torch.zeros_like(tok, dtype=torch.bool)
    mask[..., :PROMPT_DIM] = True
    tok.grad_mask = mask
    return tok

def fetta_adapt(prompt, model, dl):
    prompt = prompt.to(DEVICE).requires_grad_(True)
    opt = optim.Adam([prompt], lr=LR_PROMPT)
    model.eval()
    for _ in range(FETTA_STEPS):
        for b in dl:
            img = b["image"].to(DEVICE)
            ids = b["input_ids"].to(DEVICE)
            att = b["attention_mask"].to(DEVICE)

            with torch.no_grad():
                v = model.get_image_embeddings(img)

            B = img.size(0)
            p_tok  = prompt.expand(B,-1,-1)
            txtEmb = model.text_encoder.embeddings(ids)
            txtEmb = torch.cat([p_tok,txtEmb],1)
            t      = model.text_encoder.forward_from_emb(txtEmb, att)

            loss = entropy_loss(v,t)
            opt.zero_grad(); loss.backward()
            prompt.grad *= prompt.grad_mask.float()
            opt.step()
    return prompt.detach()

# =========================================================
# 5) 评估
# =========================================================
@torch.no_grad()
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

@torch.no_grad()
def evaluate(model, prompt, dl):
    img_all, txt_all = [], []
    for b in tqdm(dl, desc="Eval"):
        img_emb = model.get_image_embeddings(b["image"].to(DEVICE))
        img_all.append(img_emb.cpu())

        p_tok = prompt.expand(img_emb.size(0), -1, -1)
        txtEmb= model.text_encoder.embeddings(b["input_ids"].to(DEVICE))
        txtEmb= torch.cat([p_tok, txtEmb],1)
        txtFeat= model.text_encoder.forward_from_emb(txtEmb,
                                                     b["attention_mask"].to(DEVICE))
        txt_all.append(txtFeat.cpu())

    v = torch.cat(img_all,0); t = torch.cat(txt_all,0)
    return compute_metrics(v,t)

# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dl    = build_loader()
    model = load_backbone().to(DEVICE)

    print(">> Stage-1  TENT  BN-adapt")
    model = tent_adapt(model, dl)

    # ---- 自动推断嵌入维度 ----
    with torch.no_grad():
        sample_dim = model.get_image_embeddings(next(iter(dl))["image"].to(DEVICE)).size(1)

    print(">> Stage-2  FETTA prompt-adapt")
    prompt_tok = build_prompt_token(sample_dim)
    prompt_tok = fetta_adapt(prompt_tok, model, dl)

    print(">> Evaluation")
    res = evaluate(model, prompt_tok, dl); print(res)
    json.dump(res, open(os.path.join(OUTPUT_DIR,"metrics.json"),"w"), indent=2)

if __name__ == "__main__":
    main()
