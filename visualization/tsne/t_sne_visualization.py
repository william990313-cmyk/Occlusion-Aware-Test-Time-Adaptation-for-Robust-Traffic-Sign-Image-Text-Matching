#!/usr/bin/env python3
# ------------------------------------------------------------
#  t-SNE 嵌入可视化脚本 - 对比TENT前后的嵌入空间变化
# ------------------------------------------------------------
#  1. 加载预训练模型
#  2. 提取TENT前后的图像和文本嵌入
#  3. 使用t-SNE进行降维
#  4. 绘制并保存t-SNE可视化图
# ------------------------------------------------------------
import os, json, torch, math, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import copy
from typing import Dict, List, Tuple, Optional

# ================= 配置参数 =================
class Config:
    # 路径配置（用户需要修改这些路径）
    MODEL_PATH = "checkpoints/model_best.pth.tar"  # 修改为实际模型路径
    DATA_DIR = "data/val_images"         # 修改为实际数据路径
    META_PATH = "data/val_metadata.json"  # 修改为实际元数据路径
    OUTPUT_DIR = "./t_sne_results"
    
    # 设备配置
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 数据配置
    BATCH_SIZE = 64 # 提取嵌入时可以使用较大的批量
    NUM_WORKERS = 4
    NUM_SAMPLES = 500  # 用于t-SNE可视化的样本数量，不宜过大
    
    # TENT配置
    TENT_STEPS = 10
    TENT_LR = 1e-5
    
    # t-SNE配置
    TSNE_N_COMPONENTS = 2  # 降维到2维
    TSNE_PERPLEXITY = 30   # t-SNE参数，建议5到50
    TSNE_N_ITER = 1000     # 迭代次数
    TSNE_RANDOM_STATE = 42 # 随机种子，保证结果可复现

# ===========================================================
# 1) 数据集类 (与之前相同)
# ===========================================================
class ValDataset(Dataset):
    def __init__(self, root, meta_json):
        with open(meta_json, 'r') as f:
            self.meta = json.load(f)
        self.root = root
        self.tf = T.Compose([
            T.Resize((224, 224), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        img_path = os.path.join(self.root, os.path.basename(item["image_path"]))
        
        if not os.path.exists(img_path):
            img_path = item["image_path"]
        
        img = self.tf(Image.open(img_path).convert("RGB"))
        cap = item.get("caption", item.get("captions", [""])[0])

        tok = self.tok(cap, padding='max_length', truncation=True,
                       max_length=128, return_tensors='pt')
        return {
            "image": img,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "caption": cap,
            "image_path": img_path
        }

# ===========================================================
# 2) 模型加载和TENT适应 (与之前相同)
# ===========================================================
def load_backbone(model_path: str):
    try:
        from model_architecture import create_model
        
        ckpt = torch.load(model_path, map_location="cpu")
        args = ckpt.get("args", {})
        
        model = create_model(
            embed_dim=args.get("embed_dim", 512),
            image_model_type=args.get("image_model", "resnext50_32x4d"),
            text_model_type=args.get("text_model", "distilbert")
        )
        
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)
        
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保model_architecture.py文件存在且模型路径正确")
        raise

def freeze_except_bn(model):
    for p in model.parameters():
        p.requires_grad_(False)
    
    for mod in model.image_encoder.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)

def entropy_loss(v, t):
    p = (v @ t.T).softmax(1)
    return (-p * (p.log() + 1e-12)).sum(1).mean()

def tent_adapt(model, dataloader, config: Config):
    if config.TENT_STEPS == 0:
        model.eval()
        return model
    
    freeze_except_bn(model)
    model.train()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TENT_LR
    )
    
    print(f"开始TENT适应，步数: {config.TENT_STEPS}")
    for step in range(config.TENT_STEPS):
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            v = model.get_image_embeddings(batch["image"].to(config.DEVICE))
            t = model.get_text_embeddings(
                batch["input_ids"].to(config.DEVICE),
                batch["attention_mask"].to(config.DEVICE)
            )
            
            loss = entropy_loss(v, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"TENT步骤 {step+1}/{config.TENT_STEPS}, 平均损失: {avg_loss:.6f}")
    
    model.eval()
    return model

# ===========================================================
# 3) 嵌入提取函数
# ===========================================================
@torch.no_grad()
def extract_embeddings(model: nn.Module, dataloader: DataLoader, config: Config) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """提取图像和文本嵌入"""
    image_embeddings = []
    text_embeddings = []
    captions = []
    image_paths = []

    model.eval()
    for batch in tqdm(dataloader, desc="提取嵌入"):
        img_emb = model.get_image_embeddings(batch["image"].to(config.DEVICE)).cpu().numpy()
        txt_emb = model.get_text_embeddings(batch["input_ids"].to(config.DEVICE),
                                             batch["attention_mask"].to(config.DEVICE)).cpu().numpy()
        
        image_embeddings.append(img_emb)
        text_embeddings.append(txt_emb)
        captions.extend(batch["caption"])
        image_paths.extend(batch["image_path"])

    return np.vstack(image_embeddings), np.vstack(text_embeddings), captions, image_paths

# ===========================================================
# 4) t-SNE 可视化函数
# ===========================================================
def plot_tsne(image_embeddings: np.ndarray, text_embeddings: np.ndarray, 
              captions: List[str], title: str, save_path: str, config: Config):
    """绘制t-SNE可视化图"""
    # 合并图像和文本嵌入
    all_embeddings = np.vstack((image_embeddings, text_embeddings))
    
    # 运行t-SNE
    tsne = TSNE(n_components=config.TSNE_N_COMPONENTS,
                perplexity=config.TSNE_PERPLEXITY,
                n_iter=config.TSNE_N_ITER,
                random_state=config.TSNE_RANDOM_STATE,
                learning_rate='auto',
                init='random')
    
    print(f"运行t-SNE降维，样本数: {all_embeddings.shape[0]}")
    tsne_results = tsne.fit_transform(all_embeddings)
    
    # 分离图像和文本的t-SNE结果
    img_tsne = tsne_results[:len(image_embeddings)]
    txt_tsne = tsne_results[len(image_embeddings):]
    
    plt.figure(figsize=(12, 12))
    sns.scatterplot(
        x=img_tsne[:, 0], y=img_tsne[:, 1],
        hue=captions, # 可以根据caption进行着色，但样本多时会很乱
        palette=sns.color_palette("hsv", len(captions)),
        legend=False, # 默认不显示图例，因为样本太多
        alpha=0.7,
        s=50, # 图像点大小
        marker='o', label='Image Embeddings'
    )
    sns.scatterplot(
        x=txt_tsne[:, 0], y=txt_tsne[:, 1],
        hue=captions, # 可以根据caption进行着色
        palette=sns.color_palette("hsv", len(captions)),
        legend=False, # 默认不显示图例
        alpha=0.7,
        s=50, # 文本点大小
        marker='x', label='Text Embeddings'
    )
    
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # 尝试添加一些文本标签，但样本多时会非常混乱
    # for i, txt in enumerate(captions):
    #     plt.annotate(txt, (img_tsne[i, 0], img_tsne[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE图已保存到: {save_path}")

# ===========================================================
# 5) 主函数
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
        # 限制数据集大小，只取前NUM_SAMPLES个样本用于t-SNE
        if len(dataset) > config.NUM_SAMPLES:
            dataset.meta = dataset.meta[:config.NUM_SAMPLES]
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )
        print(f"成功加载数据集，共 {len(dataset)} 个样本用于t-SNE可视化")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 加载模型
    try:
        print(">>> 加载预训练模型")
        original_model = load_backbone(config.MODEL_PATH).to(config.DEVICE)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # --- 阶段1: TENT前的嵌入 --- #
    print(">>> 提取TENT前的嵌入")
    img_emb_before, txt_emb_before, captions_before, _ = extract_embeddings(original_model, dataloader, config)
    plot_tsne(img_emb_before, txt_emb_before, captions_before, 
              "t-SNE Visualization Before TENT Adaptation", 
              os.path.join(config.OUTPUT_DIR, "tsne_before_tent.png"), config)
    
    # --- 阶段2: TENT后的嵌入 --- #
    print(">>> 进行TENT适应")
    model_after_tent = copy.deepcopy(original_model) # 复制模型以进行TENT适应，不影响原始模型
    model_after_tent = tent_adapt(model_after_tent, dataloader, config)
    
    print(">>> 提取TENT后的嵌入")
    img_emb_after, txt_emb_after, captions_after, _ = extract_embeddings(model_after_tent, dataloader, config)
    plot_tsne(img_emb_after, txt_emb_after, captions_after, 
              "t-SNE Visualization After TENT Adaptation", 
              os.path.join(config.OUTPUT_DIR, "tsne_after_tent.png"), config)

    # --- 阶段3: CE-ReRank后的嵌入 (概念性，需要进一步讨论) --- #
    # CE-ReRank不直接产生新的嵌入，它是在现有嵌入相似度基础上进行重排。
    # 如果要可视化CE-ReRank的效果，通常是可视化双塔模型的嵌入，并根据CE-ReRank的结果进行颜色或形状的标注。
    # 例如，可以根据样本是否被CE-ReRank正确匹配来着色。
    # 这需要更复杂的逻辑来获取CE-ReRank的匹配结果，并将其映射回t-SNE图。
    # 暂时不实现此部分，如果需要，请进一步说明您的可视化需求。
    print(">>> CE-ReRank后的嵌入可视化需要进一步的需求明确，暂时跳过此部分。")

    print(">>> t-SNE可视化完成！结果保存在: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()

