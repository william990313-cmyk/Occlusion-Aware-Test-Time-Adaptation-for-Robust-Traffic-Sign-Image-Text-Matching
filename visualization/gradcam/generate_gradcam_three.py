#!/usr/bin/env python3
import types, torch
torch._custom_ops = types.ModuleType("torch._custom_ops")  # 临时补丁，避免 torchvision 旧/新不匹配报错

import os, cv2
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from model_architecture import create_model  # 你工程里的构造函数

# ========== 需要你改的三项 ==========
CKPT_PATH      = "checkpoints/model_best.pth.tar"        # baseline 权重
IMG_PATH       = "data/examples/fire_S_xy_049_049.jpg"      # 输入遮挡图
BACKBONE_EXPR  = "model.image_encoder.backbone"           # backbone 路径表达式
TARGET_LAYER   = "layer4"                                 # 例如 resnext 的最后 block
# ======================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 预处理 / 反归一化
to_tensor = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
denorm = T.Normalize(
    mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
    std=[1/s for s in [0.229,0.224,0.225]]
)

# 加载模型，先 strict=True, 失败再降级 strict=False（可视化场景安全）
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = create_model().to(DEVICE)
    state = ckpt.get("state_dict", ckpt)
    try:
        model.load_state_dict(state, strict=True)
        print("[Load] checkpoint loaded with strict=True")
    except Exception as e:
        print("[Warn] strict=True failed, fallback to strict=False:", e)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Load Loose] missing_keys: {missing}")
        print(f"[Load Loose] unexpected_keys: {unexpected}")
    model.eval()
    return model

# 只过 backbone 前向(避免 projection/embedding 的不一致)
def forward_backbone(model, img_tensor):
    x = img_tensor
    backbone = eval(BACKBONE_EXPR)  # 比如 model.image_encoder.backbone
    for name, module in backbone.named_children():
        x = module(x)
    return x  # [B, C, H, W]

# Grad-CAM 采集
def compute_cam(model, img_tensor):
    feats = grads = None
    layer = eval(f"{BACKBONE_EXPR}.{TARGET_LAYER}")

    def fwd(_, __, out):
        nonlocal feats; feats = out  # (B,C,H,W)
    def bwd(_, __, grad_out):
        nonlocal grads; grads = grad_out[0]

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)

    # 只用 backbone 输出做 scoring
    score = forward_backbone(model, img_tensor).norm()
    model.zero_grad()
    score.backward()

    weights = grads.mean(dim=(2,3), keepdim=True)  # (1,C,1,1)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))  # (1,1,H,W)
    cam = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear",
                        align_corners=False).squeeze().cpu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove(); h2.remove()
    return cam.numpy()

def overlay(img_tensor, cam, out_path, alpha=0.35):
    img = denorm(img_tensor.squeeze()).clamp(0,1).permute(1,2,0).cpu().numpy()
    heat = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(heat, alpha, (img*255).astype("uint8"), 1-alpha, 0)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def tent_bn_adapt(model, img_tensor, steps=20, lr=1e-5):
    # 只更新 backbone 中的 BN γ/β
    for p in model.parameters(): p.requires_grad_(False)
    backbone = eval(BACKBONE_EXPR)
    for m in backbone.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    for _ in range(steps):
        loss = forward_backbone(model, img_tensor).norm()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

def main():
    os.makedirs("gradcam_out", exist_ok=True)
    img = Image.open(IMG_PATH).convert("RGB")
    img_tensor = to_tensor(img).unsqueeze(0).to(DEVICE)

    # Baseline
    model_base = load_model()
    cam_base = compute_cam(model_base, img_tensor)
    overlay(img_tensor, cam_base, "gradcam_out/baseline.png")

    # TENT-20
    model_tent = load_model()
    tent_bn_adapt(model_tent, img_tensor, steps=20, lr=1e-5)
    cam_tent = compute_cam(model_tent, img_tensor)
    overlay(img_tensor, cam_tent, "gradcam_out/tent20.png")

    # OATA 图像分支和 TENT-20 相同，复制一份
    os.makedirs("gradcam_out", exist_ok=True)
    os.system("cp gradcam_out/tent20.png gradcam_out/oata.png")

    print("Saved: gradcam_out/baseline.png, tent20.png, oata.png")

if __name__ == "__main__":
    main()
