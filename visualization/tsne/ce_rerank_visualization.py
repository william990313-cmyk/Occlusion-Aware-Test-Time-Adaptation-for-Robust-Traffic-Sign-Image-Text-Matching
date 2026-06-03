#!/usr/bin/env python3
# ------------------------------------------------------------
#  CE-ReRank 效果可视化脚本
# ------------------------------------------------------------
#  1. 从评估脚本中提取原始排名和重排后的排名
#  2. 可视化排名变化
#  3. 可视化相似度分数分布
#  4. 可视化Top-K匹配示例
# ------------------------------------------------------------
import os, json, torch, math, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from typing import Dict, List, Tuple

# ================= 配置参数 =================
class Config:
    # 路径配置（用户需要修改这些路径）
    MODEL_PATH = "checkpoints/model_best.pth.tar"  # 修改为实际模型路径
    DATA_DIR   = "data/val_images"         # 修改为实际数据路径
    META_PATH  = "data/val_metadata.json"  # 修改为实际元数据路径
    OUTPUT_DIR = "./ce_rerank_vis_results"
    
    # 设备配置
    DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 数据配置
    BATCH_SIZE = 64
    NUM_WORKERS= 4
    
    # TENT配置
    TENT_STEPS = 10
    TENT_LR    = 1e-5
    
    # Cross-Encoder Re-Rank 配置
    TOPK       = 64
    CE_ID      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # 可视化配置
    NUM_SAMPLES_TO_VISUALIZE = 20 # 用于排名变化可视化的样本数量
    FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc" # 中文字体路径

# ===========================================================
# 1) 数据集 (与 evaluate_occlusion_tent_CE_ReRank.py 相同)
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
            "caption": cap,
            "image_path": img_path # 添加image_path方便后续可视化
        }

def build_loader(config: Config):
    return DataLoader(ValDataset(config.DATA_DIR, config.META_PATH),
                      batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS)

# ===========================================================
# 2) 双塔模型 (与 evaluate_occlusion_tent_CE_ReRank.py 相同)
# ===========================================================
from model_architecture import create_model          # ← 如名称不同请改

def load_backbone(config: Config):
    ckpt  = torch.load(config.MODEL_PATH, map_location="cpu")
    args  = ckpt.get("args", {})
    model = create_model(embed_dim=args.get("embed_dim",512),
                         image_model_type=args.get("image_model","resnet50"),
                         text_model_type=args.get("text_model","distilbert"))
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model

# ===========================================================
# 3) TENT 适应 (与 evaluate_occlusion_tent_CE_ReRank.py 相同)
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

def tent_adapt(model, dl, config: Config):
    if config.TENT_STEPS == 0:
        model.eval(); return model
    freeze_except_bn(model); model.train()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=config.TENT_LR)
    for _ in range(config.TENT_STEPS):
        for batch in dl:
            v = model.get_image_embeddings(batch["image"].to(config.DEVICE))
            t = model.get_text_embeddings(batch["input_ids"].to(config.DEVICE),
                                          batch["attention_mask"].to(config.DEVICE))
            loss = entropy_loss(v, t)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

# ===========================================================
# 4) 交叉编码器 (与 evaluate_occlusion_tent_CE_ReRank.py 相同)
# ===========================================================
ce_tok_global   = None
ce_model_global = None

def init_ce_model(config: Config):
    global ce_tok_global, ce_model_global
    if ce_tok_global is None:
        ce_tok_global   = AutoTokenizer.from_pretrained(config.CE_ID)
        ce_model_global = AutoModelForSequenceClassification.from_pretrained(config.CE_ID).to(config.DEVICE)
        ce_model_global.eval()

@torch.no_grad()
def ce_score(text_pairs, config: Config):
    init_ce_model(config)
    tok = ce_tok_global(text_pairs[0], text_pairs[1],
                 padding=True, truncation=True,
                 max_length=128, return_tensors='pt').to(config.DEVICE)
    return ce_model_global(**tok).logits.squeeze(-1)

# ===========================================================
# 5) 指标 + Re-Rank (修改以返回原始排名和重排后的排名)
# ===========================================================
@torch.no_grad()
def compute_ranks(dual_img, dual_txt, captions, config: Config):
    sim   = dual_img @ dual_txt.T
    N     = sim.size(0)
    rank0 = sim.argsort(1, descending=True)             # (N,N) 原始排名

    new_rank = torch.empty_like(rank0)
    for i in tqdm(range(N), desc="Cross-Encoder Re-Ranking"):
        top_idx = rank0[i, :config.TOPK]
        pair1   = [captions[i]] * config.TOPK
        pair2   = [captions[j] for j in top_idx.tolist()]
        scores  = ce_score((pair1, pair2), config).cpu()
        _, rer  = scores.sort(descending=True)
        new_rank[i] = torch.cat([top_idx[rer], rank0[i, config.TOPK:]], 0)

    return rank0, new_rank, sim, scores # 返回原始排名、重排后的排名、双塔相似度、CE相似度

# ===========================================================
# 6) 可视化函数
# ===========================================================
def plot_rank_change(original_ranks: torch.Tensor, reranked_ranks: torch.Tensor,
                     captions: List[str], image_paths: List[str], config: Config):
    """可视化排名的变化"""
    N = original_ranks.size(0)
    
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"] # 或者其他支持中文的字体，如 "Noto Sans CJK SC"
    plt.rcParams["axes.unicode_minus"] = False

    # 随机选择一部分样本进行可视化
    indices = torch.randperm(N)[:config.NUM_SAMPLES_TO_VISUALIZE]
    
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, 4 * len(indices)))
    if len(indices) == 1: # 如果只有一个子图，axes不是数组
        axes = [axes]

    for i, idx in enumerate(indices):
        original_pos = (original_ranks[idx] == idx).nonzero(as_tuple=True)[0].item()
        reranked_pos = (reranked_ranks[idx] == idx).nonzero(as_tuple=True)[0].item()
        
        ax = axes[i]
        ax.bar(["Original Rank", "Re-ranked Rank"], [original_pos + 1, reranked_pos + 1], 
               color=["skyblue", "lightcoral"])
        ax.set_ylabel("Rank Position")
        ax.set_title(f"Sample {idx}: Rank Change (Caption: {captions[idx][:30]}...)")
        ax.text(0, original_pos + 1, f" {original_pos + 1}", ha="center", va="bottom")
        ax.text(1, reranked_pos + 1, f" {reranked_pos + 1}", ha="center", va="bottom")
        
        # 突出显示排名提升或下降
        if reranked_pos < original_pos:
            ax.patch.set_facecolor("lightgreen") # 排名提升
        elif reranked_pos > original_pos:
            ax.patch.set_facecolor("lightsalmon") # 排名下降
        else:
            ax.patch.set_facecolor("lightgray") # 排名不变

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "rank_change_visualization.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"排名变化可视化图已保存到: {save_path}")

def plot_similarity_distribution(dual_encoder_sim: torch.Tensor, ce_rerank_sim: torch.Tensor,
                                 config: Config):
    """可视化相似度分数分布对比"""
    plt.figure(figsize=(12, 6))
    sns.histplot(dual_encoder_sim.flatten().cpu().numpy(), color="skyblue", label="Dual-Encoder Similarity", kde=True, stat="density", alpha=0.5)
    sns.histplot(ce_rerank_sim.flatten().cpu().numpy(), color="lightcoral", label="CE-ReRank Similarity", kde=True, stat="density", alpha=0.5)
    plt.title("Similarity Score Distribution Comparison")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    save_path = os.path.join(config.OUTPUT_DIR, "similarity_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"相似度分数分布图已保存到: {save_path}")

def visualize_topk_matches(model, dataset, original_ranks, reranked_ranks, config: Config, num_examples=5):
    """可视化Top-K匹配示例"""
    # 随机选择一些样本
    indices = torch.randperm(len(dataset))[:num_examples]

    for i, idx in enumerate(indices):
        query_image_path = dataset[idx]["image_path"]
        query_caption = dataset[idx]["caption"]

        # 获取原始Top-K和重排Top-K的索引
        original_topk_indices = original_ranks[idx, :config.TOPK].tolist()
        reranked_topk_indices = reranked_ranks[idx, :config.TOPK].tolist()

        fig, axes = plt.subplots(2, config.TOPK + 1, figsize=(2 * (config.TOPK + 1), 6))
        
        # 显示查询图像
        query_img = Image.open(query_image_path).convert("RGB")
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title(f"Query ({idx})\n{query_caption[:20]}...")
        axes[0, 0].axis("off")
        axes[1, 0].imshow(query_img)
        axes[1, 0].set_title(f"Query ({idx})\n{query_caption[:20]}...")
        axes[1, 0].axis("off")

        # 显示原始Top-K匹配
        for j, match_idx in enumerate(original_topk_indices):
            match_image_path = dataset[match_idx]["image_path"]
            match_caption = dataset[match_idx]["caption"]
            img = Image.open(match_image_path).convert("RGB")
            axes[0, j+1].imshow(img)
            axes[0, j+1].set_title(f"Rank {j+1} ({match_idx})\n{match_caption[:20]}...")
            axes[0, j+1].axis("off")
            if match_idx == idx: # 如果是正确匹配
                axes[0, j+1].patch.set_edgecolor("green")
                axes[0, j+1].patch.set_linewidth(3)

        # 显示重排Top-K匹配
        for j, match_idx in enumerate(reranked_topk_indices):
            match_image_path = dataset[match_idx]["image_path"]
            match_caption = dataset[match_idx]["caption"]
            img = Image.open(match_image_path).convert("RGB")
            axes[1, j+1].imshow(img)
            axes[1, j+1].set_title(f"Rank {j+1} ({match_idx})\n{match_caption[:20]}...")
            axes[1, j+1].axis("off")
            if match_idx == idx: # 如果是正确匹配
                axes[1, j+1].patch.set_edgecolor("green")
                axes[1, j+1].patch.set_linewidth(3)

        fig.suptitle(f"Top-{config.TOPK} Matches for Sample {idx}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(config.OUTPUT_DIR, f"topk_matches_sample_{idx}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Top-{config.TOPK}匹配示例图已保存到: {save_path}")

# ===========================================================
# 7) 主函数
# ===========================================================
def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 检查配置
    if not os.path.exists(config.MODEL_PATH):
        print(f"错误: 模型文件不存在: {config.MODEL_PATH}")
        print("请修改Config类中的MODEL_PATH为正确的模型路径")
        return
    
    if not os.path.exists(config.DATA_DIR):
        print(f"错误: 数据目录不存在: {config.DATA_DIR}")
        print("请修改Config类中的DATA_DIR为正确的数据路径")
        return
    
    if not os.path.exists(config.META_PATH):
        print(f"错误: 元数据文件不存在: {config.META_PATH}")
        print("请修改Config类中的META_PATH为正确的元数据路径")
        return
    
    # 创建数据加载器
    try:
        dataset = ValDataset(config.DATA_DIR, config.META_PATH)
        dataloader = build_loader(config)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 加载模型
    try:
        print(">>> 加载预训练模型")
        model = load_backbone(config).to(config.DEVICE)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # TENT 适应
    print(">>> Stage-1  TENT BN-adapt")
    model = tent_adapt(model, dataloader, config)

    # 提取双塔模型嵌入
    print(">>> Stage-2  Dual-Encoder forward")
    img_emb, txt_emb, caps, img_paths = [], [], [], []
    with torch.no_grad():
        for b in tqdm(dataloader, desc="DualEmb"):
            img_emb.append(model.get_image_embeddings(b["image"].to(config.DEVICE)).cpu())
            txt_emb.append(model.get_text_embeddings(b["input_ids"].to(config.DEVICE),
                                                     b["attention_mask"].to(config.DEVICE)).cpu())
            caps.extend(b["caption"])
            img_paths.extend(b["image_path"])
    img_emb = torch.cat(img_emb, 0)
    txt_emb = torch.cat(txt_emb, 0)

    # 计算原始排名和重排后的排名
    print(">>> Stage-3  Compute Ranks with Cross-Encoder Re-Rank")
    original_ranks, reranked_ranks, dual_encoder_sim_scores, ce_rerank_sim_scores = compute_ranks(img_emb, txt_emb, caps, config)
    
    # --- 可视化 ---
    print(">>> 开始可视化")
    
    # 方案1: 排名变化可视化
    plot_rank_change(original_ranks, reranked_ranks, caps, img_paths, config)
    
    # 方案2: 相似度分数分布对比
    plot_similarity_distribution(dual_encoder_sim_scores, ce_rerank_sim_scores, config)
    
    # 方案4: Top-K匹配示例对比
    visualize_topk_matches(model, dataset, original_ranks, reranked_ranks, config, num_examples=3)

    print(">>> CE-ReRank可视化完成！结果保存在: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()

