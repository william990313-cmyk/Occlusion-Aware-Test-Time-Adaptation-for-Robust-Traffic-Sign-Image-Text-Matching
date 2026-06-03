#!/usr/bin/env python3
# ------------------------------------------------------------
#  综合热区图分析脚本 - 对比TENT前后的视觉处理效果
# ------------------------------------------------------------
#  1. 支持多种热区图生成方法
#  2. 生成详细的对比分析
#  3. 包含统计分析和可视化
#  4. 支持批量处理和结果保存
# ------------------------------------------------------------
import os, json, torch, math, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_grad_cam import EnhancedGradCAM, show_cam_on_image, create_heatmap_grid
import copy
from typing import Dict, List, Tuple, Optional

# ================= 配置参数 =================
class Config:
    # 路径配置（用户需要修改这些路径）
    MODEL_PATH = "checkpoints/model_best.pth.tar"  # 修改为实际模型路径
    DATA_DIR = "data/val_images"         # 修改为实际数据路径
    META_PATH = "data/val_metadata.json"  # 修改为实际元数据路径
    OUTPUT_DIR = "./comprehensive_heatmap_results"
    
    # 设备配置
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 数据配置
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    NUM_SAMPLES = 10  # 分析的样本数量
    
    # TENT配置
    TENT_STEPS = 10
    TENT_LR = 1e-5
    
    # 热区图配置
    TARGET_LAYERS = [
        "image_encoder.backbone.layer4",
        "image_encoder.backbone.layer3", 
        "image_encoder.backbone.layer2"
    ]
    HEATMAP_METHODS = ["embedding_sum", "similarity"]
    
    # 可视化配置
    COLORMAP = cv2.COLORMAP_JET
    ALPHA = 0.4

# ===========================================================
# 1) 数据集类
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
        
        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            # 如果文件不存在，尝试使用原始路径
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
# 2) 模型加载和TENT适应
# ===========================================================
def load_backbone(model_path: str):
    """加载预训练模型"""
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
    """冻结除BN层外的所有参数"""
    for p in model.parameters():
        p.requires_grad_(False)
    
    for mod in model.image_encoder.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)

def entropy_loss(v, t):
    """计算熵损失"""
    p = (v @ t.T).softmax(1)
    return (-p * (p.log() + 1e-12)).sum(1).mean()

def tent_adapt(model, dataloader, config: Config):
    """TENT适应"""
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
# 3) 热区图分析类
# ===========================================================
class HeatmapAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.results = {
            'samples': [],
            'statistics': {},
            'config': {
                'tent_steps': config.TENT_STEPS,
                'tent_lr': config.TENT_LR,
                'num_samples': config.NUM_SAMPLES,
                'target_layers': config.TARGET_LAYERS,
                'methods': config.HEATMAP_METHODS
            }
        }

    def analyze_sample(self, model_before: nn.Module, model_after: nn.Module,
                      sample_data: Dict, sample_idx: int) -> Dict:
        """分析单个样本的热区图"""
        image = sample_data['image']
        caption = sample_data['caption']
        input_ids = sample_data['input_ids']
        attention_mask = sample_data['attention_mask']
        
        sample_result = {
            'index': sample_idx,
            'caption': caption,
            'image_path': sample_data['image_path'],
            'heatmaps': {}
        }
        
        # 为每个目标层和方法生成热区图
        for layer_name in self.config.TARGET_LAYERS:
            sample_result['heatmaps'][layer_name] = {}
            
            for method in self.config.HEATMAP_METHODS:
                try:
                    # TENT前的热区图
                    grad_cam_before = EnhancedGradCAM(model_before, layer_name)
                    if method == "similarity":
                        heatmap_before = grad_cam_before.generate_heatmap(
                            image.unsqueeze(0), 
                            input_ids.unsqueeze(0),
                            attention_mask.unsqueeze(0),
                            method=method
                        )
                    else:
                        heatmap_before = grad_cam_before.generate_heatmap(
                            image.unsqueeze(0), method=method
                        )
                    grad_cam_before.cleanup()
                    
                    # TENT后的热区图
                    grad_cam_after = EnhancedGradCAM(model_after, layer_name)
                    if method == "similarity":
                        heatmap_after = grad_cam_after.generate_heatmap(
                            image.unsqueeze(0),
                            input_ids.unsqueeze(0),
                            attention_mask.unsqueeze(0),
                            method=method
                        )
                    else:
                        heatmap_after = grad_cam_after.generate_heatmap(
                            image.unsqueeze(0), method=method
                        )
                    grad_cam_after.cleanup()
                    
                    # 计算热区图差异
                    diff_metrics = self.calculate_heatmap_difference(
                        heatmap_before, heatmap_after
                    )
                    
                    sample_result['heatmaps'][layer_name][method] = {
                        'before': heatmap_before.tolist(),
                        'after': heatmap_after.tolist(),
                        'difference_metrics': diff_metrics
                    }
                    
                except Exception as e:
                    print(f"生成热区图失败 - 层: {layer_name}, 方法: {method}, 错误: {e}")
                    sample_result['heatmaps'][layer_name][method] = None
        
        return sample_result

    def calculate_heatmap_difference(self, heatmap1: np.ndarray, 
                                   heatmap2: np.ndarray) -> Dict:
        """计算两个热区图之间的差异指标"""
        # 确保热区图大小一致
        if heatmap1.shape != heatmap2.shape:
            heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))
        
        # 计算各种差异指标
        mse = np.mean((heatmap1 - heatmap2) ** 2)
        mae = np.mean(np.abs(heatmap1 - heatmap2))
        
        # 结构相似性指数（简化版）
        def ssim_simple(img1, img2):
            mu1, mu2 = img1.mean(), img2.mean()
            sigma1, sigma2 = img1.var(), img2.var()
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
            return ssim
        
        ssim = ssim_simple(heatmap1, heatmap2)
        
        # 相关系数
        correlation = np.corrcoef(heatmap1.flatten(), heatmap2.flatten())[0, 1]
        
        # 热区重叠度（阈值化后的交集比并集）
        threshold = 0.5
        binary1 = (heatmap1 > threshold).astype(float)
        binary2 = (heatmap2 > threshold).astype(float)
        intersection = np.sum(binary1 * binary2)
        union = np.sum(np.maximum(binary1, binary2))
        iou = intersection / (union + 1e-8)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'ssim': float(ssim),
            'correlation': float(correlation),
            'iou': float(iou)
        }

    def save_visualization(self, sample_data: Dict, sample_result: Dict, 
                          output_dir: str) -> List[str]:
        """保存可视化结果"""
        saved_paths = []
        sample_idx = sample_result['index']
        image = sample_data['image']
        
        for layer_name in self.config.TARGET_LAYERS:
            for method in self.config.HEATMAP_METHODS:
                heatmap_data = sample_result['heatmaps'][layer_name].get(method)
                if heatmap_data is None:
                    continue
                
                heatmap_before = np.array(heatmap_data['before'])
                heatmap_after = np.array(heatmap_data['after'])
                
                # 创建可视化
                original_img = show_cam_on_image(image, np.zeros_like(heatmap_before), alpha=0)
                heatmap_before_vis = show_cam_on_image(image, heatmap_before, self.config.ALPHA)
                heatmap_after_vis = show_cam_on_image(image, heatmap_after, self.config.ALPHA)
                
                # 创建差异图
                diff_heatmap = np.abs(heatmap_after - heatmap_before)
                diff_vis = show_cam_on_image(image, diff_heatmap, self.config.ALPHA, cv2.COLORMAP_HOT)
                
                # 创建网格显示
                images = [original_img, heatmap_before_vis, heatmap_after_vis, diff_vis]
                titles = ['Original', 'Before TENT', 'After TENT', 'Difference']
                
                grid_img = create_heatmap_grid(images, titles, figsize=(20, 5))
                
                # 保存图像
                filename = f'sample_{sample_idx}_{layer_name.replace(".", "_")}_{method}.png'
                save_path = os.path.join(output_dir, filename)
                
                plt.figure(figsize=(20, 5))
                plt.imshow(grid_img)
                plt.axis('off')
                plt.title(f'Sample {sample_idx} - {layer_name} - {method}\n{sample_result["caption"][:100]}...')
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                saved_paths.append(save_path)
        
        return saved_paths

    def generate_statistics(self):
        """生成统计分析"""
        if not self.results['samples']:
            return
        
        # 收集所有差异指标
        all_metrics = {metric: [] for metric in ['mse', 'mae', 'ssim', 'correlation', 'iou']}
        
        for sample in self.results['samples']:
            for layer_name in sample['heatmaps']:
                for method in sample['heatmaps'][layer_name]:
                    heatmap_data = sample['heatmaps'][layer_name][method]
                    if heatmap_data and 'difference_metrics' in heatmap_data:
                        metrics = heatmap_data['difference_metrics']
                        for metric_name, value in metrics.items():
                            if not np.isnan(value):
                                all_metrics[metric_name].append(value)
        
        # 计算统计量
        statistics = {}
        for metric_name, values in all_metrics.items():
            if values:
                statistics[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        self.results['statistics'] = statistics

    def save_results(self, output_dir: str):
        """保存分析结果"""
        # 生成统计分析
        self.generate_statistics()
        
        # 保存JSON结果
        results_path = os.path.join(output_dir, 'analysis_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 生成统计图表
        self.plot_statistics(output_dir)
        
        print(f"分析结果已保存到: {results_path}")

    def plot_statistics(self, output_dir: str):
        """绘制统计图表"""
        if not self.results['statistics']:
            return
        
        # 创建统计图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = list(self.results['statistics'].keys())
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
            
            stats = self.results['statistics'][metric]
            values = [stats['min'], stats['mean'], stats['max']]
            labels = ['Min', 'Mean', 'Max']
            
            axes[i].bar(labels, values)
            axes[i].set_title(f'{metric.upper()} Statistics')
            axes[i].set_ylabel('Value')
        
        # 隐藏多余的子图
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'statistics_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ===========================================================
# 4) 主函数
# ===========================================================
def main():
    config = Config()
    
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
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 创建数据加载器
    try:
        dataset = ValDataset(config.DATA_DIR, config.META_PATH)
        dataloader = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
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
    
    # 创建TENT前的模型副本
    print(">>> 创建模型副本用于对比")
    model_before_tent = copy.deepcopy(original_model)
    model_before_tent.eval()
    
    # 进行TENT适应
    print(">>> 开始TENT适应")
    model_after_tent = tent_adapt(original_model, dataloader, config)
    
    # 创建热区图分析器
    analyzer = HeatmapAnalyzer(config)
    
    # 分析样本
    print(f">>> 开始分析 {config.NUM_SAMPLES} 个样本")
    sample_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="分析样本")):
        if sample_count >= config.NUM_SAMPLES:
            break
        
        sample_data = {
            'image': batch['image'][0],
            'caption': batch['caption'][0],
            'input_ids': batch['input_ids'][0],
            'attention_mask': batch['attention_mask'][0],
            'image_path': batch['image_path'][0]
        }
        
        try:
            # 分析样本
            sample_result = analyzer.analyze_sample(
                model_before_tent, model_after_tent, sample_data, sample_count
            )
            
            # 保存可视化
            saved_paths = analyzer.save_visualization(
                sample_data, sample_result, config.OUTPUT_DIR
            )
            sample_result['visualization_paths'] = saved_paths
            
            analyzer.results['samples'].append(sample_result)
            sample_count += 1
            
            print(f"完成样本 {sample_count}/{config.NUM_SAMPLES}")
            
        except Exception as e:
            print(f"分析样本 {sample_count} 时出错: {e}")
            continue
    
    # 保存最终结果
    print(">>> 保存分析结果")
    analyzer.save_results(config.OUTPUT_DIR)
    
    print(f">>> 分析完成！结果保存在: {config.OUTPUT_DIR}")
    print(f">>> 成功分析了 {len(analyzer.results['samples'])} 个样本")
    
    # 打印统计摘要
    if analyzer.results['statistics']:
        print("\n=== 统计摘要 ===")
        for metric, stats in analyzer.results['statistics'].items():
            print(f"{metric.upper()}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")

if __name__ == "__main__":
    main()

