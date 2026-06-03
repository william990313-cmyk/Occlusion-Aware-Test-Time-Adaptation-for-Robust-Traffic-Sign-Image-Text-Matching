# ------------------------------------------------------------
#  Pure TENT  +  ROID  (outlier-dampened BN 更新)
# ------------------------------------------------------------
import os, json, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------- 路径 ----------
MODEL_CKPT = "checkpoints/model_best.pth.tar"
IMG_DIR    = "data/val_images"
META_JSON  = "data/val_metadata.json"
OUT_DIR    = "./roid_result"

DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS= 4

# ---------- TENT ----------
TENT_STEPS = 20
TENT_LR    = 1e-5

# ---------- ROID ----------
TAU_STD     = 3.0     # 异常阈值  |Δμ| > τ·σ
DAMP_FACTOR = 0.5     # 阻尼比率

# ============================================================
# 1) 数据
# ============================================================
class ValDS(Dataset):
    def __init__(self, root, meta):
        self.meta = json.load(open(meta))
        self.root = root
        self.tf   = T.Compose([
            T.Resize((224,224)),
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

def loader():
    return DataLoader(ValDS(IMG_DIR, META_JSON),
                      batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)

# ============================================================
# 2) 模型
# ============================================================
from model_architecture import create_model          # ← 依工程修改

def load_model():
    ckpt  = torch.load(MODEL_CKPT, map_location="cpu")
    args  = ckpt.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim",512),
                         image_model_type=args.get("image_model","resnet50"),
                         text_model_type=args.get("text_model","distilbert"))
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model

# ============================================================
# 3) 适应：TENT + ROID
# ============================================================
def freeze_except_bn(m):
    for p in m.parameters(): p.requires_grad_(False)
    for layer in m.image_encoder.modules():
        if isinstance(layer,(nn.BatchNorm2d, nn.BatchNorm1d)):
            layer.weight.requires_grad_(True)
            layer.bias.requires_grad_(True)

def entropy(v, t):
    p = (v @ t.T).softmax(1)
    return (-p * (p.log()+1e-12)).sum(1).mean()

def register_prev(m):
    for layer in m.modules():
        if isinstance(layer,(nn.BatchNorm2d, nn.BatchNorm1d)):
            layer.register_buffer("prev_mean", layer.running_mean.clone())
            layer.register_buffer("prev_var" , layer.running_var .clone())

@torch.no_grad()
def damp_bn(layer, tau=TAU_STD, damp=DAMP_FACTOR):
    mu  = layer.running_mean
    var = layer.running_var
    std = torch.sqrt(var + 1e-5)

    delta = mu - layer.prev_mean
    mask  = delta.abs() > tau * std
    if mask.any():
        mu_new = mu.clone()
        mu_new[mask] -= damp * delta[mask]
        layer.running_mean.copy_(mu_new)

    # 可选：对 var 做同样操作
    # vdelta = var - layer.prev_var
    # vmask  = vdelta.abs() > tau * layer.prev_var
    # if vmask.any():
    #     var_new = var.clone()
    #     var_new[vmask] -= damp * vdelta[vmask]
    #     layer.running_var.copy_(var_new)

    layer.prev_mean.copy_(layer.running_mean)
    layer.prev_var .copy_(layer.running_var)

def tent_roid(model, dl):
    freeze_except_bn(model); model.train()
    register_prev(model)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=TENT_LR)

    for _ in range(TENT_STEPS):
        for batch in dl:
            v = model.get_image_embeddings(batch["image"].to(DEVICE))
            t = model.get_text_embeddings(batch["input_ids"].to(DEVICE),
                                          batch["attention_mask"].to(DEVICE))
            loss = entropy(v,t)

            opt.zero_grad(); loss.backward(); opt.step()

            # —— ROID: 对所有 BN 阻尼 —— #
            for layer in model.modules():
                if isinstance(layer,(nn.BatchNorm2d, nn.BatchNorm1d)):
                    damp_bn(layer)
    model.eval(); return model

# ============================================================
# 4) 评估
# ============================================================
@torch.no_grad()
def metrics(m, dl):
    v_all, t_all = [], []
    for b in tqdm(dl, desc="Eval"):
        v_all.append(m.get_image_embeddings(b["image"].to(DEVICE)).cpu())
        t_all.append(m.get_text_embeddings(b["input_ids"].to(DEVICE),
                                           b["attention_mask"].to(DEVICE)).cpu())
    v_all = torch.cat(v_all,0); t_all = torch.cat(t_all,0)
    sim   = v_all @ t_all.T
    rank  = sim.argsort(1, descending=True)
    lab   = torch.arange(sim.size(0))
    r1  = (rank[:,:1]==lab[:,None]).float().mean().item()*100
    r5  = (rank[:,:5]==lab[:,None]).any(1).float().mean().item()*100
    r10 = (rank[:,:10]==lab[:,None]).any(1).float().mean().item()*100
    # mAP
    eq   = (rank==lab[:,None])
    prec = eq.float().cumsum(1) / (torch.arange(sim.size(0))+1)
    mAP  = (prec*eq.float()).sum(1).mean().item()*100
    return {"R@1":round(r1,2),"R@5":round(r5,2),
            "R@10":round(r10,2),"mAP":round(mAP,2)}

# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dl     = loader()
    model  = load_model().to(DEVICE)

    print(">> TENT + ROID adapting ...")
    model  = tent_roid(model, dl)

    print(">> Evaluating ...")
    res = metrics(model, dl);  print(res)
    json.dump(res, open(os.path.join(OUT_DIR,"metrics.json"),"w"), indent=2)

if __name__ == "__main__":
    main()
