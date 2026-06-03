import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from model_architecture import create_model
from utils import compute_metrics

# 硬编码路径参数 - 使用绝对路径，适配用户环境
MODEL_PATH = "checkpoints/model_best.pth.tar"  # 预训练模型路径
DATASET_PATH = "data/val_images"   # 遮挡数据集路径
METADATA_PATH = "data/val_metadata.json" # 文本描述JSON文件路径
OUTPUT_DIR = "./occlusion_eval_results"                   # 结果保存路径
BATCH_SIZE = 64                                           # 批次大小
NUM_WORKERS = 4                                           # 数据加载线程数
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # 设备

# TENT参数
TENT_STEPS = 20                                           # TENT适应步数
TENT_LR = 1e-05                                        # TENT学习率
USE_OCCLUSION_LOSS = True                                # 是否使用遮挡不变性对比损失
USE_ENTROPY_LOSS = True                                  # 是否使用基于熵的对比损失
TEMPERATURE = 0.3                                         # 温度参数
ENTROPY_WEIGHT = 0.9                                      # 熵损失权重

# 自定义数据集类，仅用于评估
class OcclusionDataset(Dataset):
    def __init__(self, data_dir, metadata_path, text_model_type='distilbert', image_model_type='resnet50_bs64', apply_occlusion=False):
        """
        初始化遮挡数据集
        
        参数:
            data_dir: 数据目录
            metadata_path: 元数据JSON文件路径
            text_model_type: 文本模型类型
            image_model_type: 图像模型类型
            apply_occlusion: 是否应用额外的随机遮挡（用于遮挡不变性损失）
        """
        self.data_dir = data_dir
        self.metadata_path = metadata_path  # 存储为实例变量
        self.apply_occlusion = apply_occlusion
        self.text_model_type = text_model_type  # 存储为实例变量
        self.image_model_type = image_model_type  # 存储为实例变量
        
        # 加载元数据
        print(f"加载元数据: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 设置图像预处理
        if 'efficientnet' in image_model_type:
            self.image_size = 380
        else:
            self.image_size = 224
            
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 设置文本预处理
        if text_model_type == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif text_model_type == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        else:  # distilbert
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.max_length = 128
        
        print(f"数据集初始化完成，共 {len(self.metadata)} 个样本")
    
    def apply_random_occlusion(self, image_tensor):
        """应用随机遮挡"""
        # 将张量转换为numpy数组以便处理
        image_np = image_tensor.numpy()
        
        # 随机生成遮挡区域的位置和大小
        h, w = self.image_size, self.image_size
        occlusion_size_h = random.randint(h // 8, h // 4)
        occlusion_size_w = random.randint(w // 8, w // 4)
        occlusion_x = random.randint(0, w - occlusion_size_w)
        occlusion_y = random.randint(0, h - occlusion_size_h)
        
        # 应用遮挡（设置为黑色或随机值）
        image_np[:, occlusion_y:occlusion_y+occlusion_size_h, occlusion_x:occlusion_x+occlusion_size_w] = 0
        
        # 转回张量
        return torch.from_numpy(image_np)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # 获取图像路径和类别
        image_path = item['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.data_dir, os.path.basename(image_path))
        
        category = item['category']
        
        # 获取文本描述
        caption = item.get('captions', [''])[0]  # 使用第一个描述
        if not caption and 'caption' in item:
            caption = item['caption']
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            # 如果需要应用额外的随机遮挡
            if self.apply_occlusion:
                image_occluded = self.apply_random_occlusion(image)
                # 返回原始图像和遮挡图像对
                return {
                    'image': image,
                    'image_occluded': image_occluded,
                    'input_ids': self.tokenizer(
                        caption,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )['input_ids'].squeeze(),
                    'attention_mask': self.tokenizer(
                        caption,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )['attention_mask'].squeeze(),
                    'category': category,
                    'image_path': image_path,
                    'caption': caption
                }
            
        except Exception as e:
            print(f"加载图像 {image_path} 时出错: {e}")
            # 返回空白图像
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # 处理文本
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 返回样本
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'category': category,
            'image_path': image_path,
            'caption': caption
        }

def create_occlusion_dataloader(data_dir, metadata_path, text_model_type, image_model_type, batch_size=64, num_workers=4, apply_occlusion=False):
    """
    创建遮挡数据集的DataLoader
    
    参数:
        data_dir: 数据目录
        metadata_path: 元数据JSON文件路径
        text_model_type: 文本模型类型
        image_model_type: 图像模型类型
        batch_size: 批次大小
        num_workers: 数据加载线程数
        apply_occlusion: 是否应用额外的随机遮挡
    
    返回:
        dataloader: 遮挡数据集的DataLoader
    """
    try:
        # 创建自定义数据集
        dataset = OcclusionDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            text_model_type=text_model_type,
            image_model_type=image_model_type,
            apply_occlusion=apply_occlusion
        )
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return dataloader
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        return None

def configure_model_for_tent(model):
    """
    配置模型以适用于TENT
    
    参数:
        model: 预训练模型
    
    返回:
        model: 配置后的模型
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 仅保持图像编码器中的BatchNorm层参数可训练
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for param_name, param in module.named_parameters():
                param.requires_grad = True
    
    return model

def entropy_loss(logits):
    """
    计算熵损失
    
    参数:
        logits: 模型输出的logits
    
    返回:
        loss: 熵损失
    """
    # 对logits应用softmax得到概率分布
    probs = F.softmax(logits, dim=1)
    # 计算熵: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    # 返回平均熵
    return entropy.mean()

def occlusion_invariance_loss(orig_embeds, occluded_embeds, temperature=0.1, use_entropy=False, entropy_weight=0.1):
    """
    计算遮挡不变性对比损失，可选择是否使用基于熵的对比损失
    
    参数:
        orig_embeds: 原始图像的嵌入
        occluded_embeds: 遮挡图像的嵌入
        temperature: 温度参数
        use_entropy: 是否使用基于熵的对比损失
        entropy_weight: 熵损失权重
    
    返回:
        loss: 对比损失
    """
    # 归一化嵌入
    orig_embeds = F.normalize(orig_embeds, p=2, dim=1)
    occluded_embeds = F.normalize(occluded_embeds, p=2, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(orig_embeds, occluded_embeds.t()) / temperature
    
    # 创建标签（对角线为正样本）
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    
    # 计算交叉熵损失
    contrastive_loss = F.cross_entropy(sim_matrix, labels)
    
    if use_entropy:
        # 计算相似度分布的熵
        sim_probs = F.softmax(sim_matrix, dim=1)
        distribution_entropy = -torch.sum(sim_probs * torch.log(sim_probs + 1e-10), dim=1).mean()
        
        # 结合对比损失和熵
        # 减去熵项，鼓励更多样化的特征表示
        total_loss = contrastive_loss - entropy_weight * distribution_entropy
        return total_loss
    else:
        return contrastive_loss

def apply_tent(model, test_loader, device, steps=10, lr=0.00001, use_occlusion_loss=False, use_entropy_loss=False, text_model_type='roberta', image_model_type='resnet50'):
    """
    应用TENT测试时适应
    
    参数:
        model: 预训练模型
        test_loader: 测试数据加载器
        device: 设备
        steps: 适应步数
        lr: 学习率
        use_occlusion_loss: 是否使用遮挡不变性对比损失
        use_entropy_loss: 是否使用基于熵的对比损失
        text_model_type: 文本模型类型
        image_model_type: 图像模型类型
    
    返回:
        model: 适应后的模型
    """
    # 配置模型以适用于TENT
    model = configure_model_for_tent(model)
    model.train()  # 设置为训练模式以更新BN统计量
    
    # 创建优化器，仅优化可训练参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # 如果使用遮挡不变性对比损失，需要创建带有遮挡的数据加载器
    if use_occlusion_loss:
        # 直接使用全局变量中的路径，而不是尝试从数据集对象中获取
        occlusion_loader = create_occlusion_dataloader(
            data_dir=DATASET_PATH,
            metadata_path=METADATA_PATH,
            text_model_type=text_model_type,
            image_model_type=image_model_type,
            batch_size=test_loader.batch_size,
            num_workers=test_loader.num_workers,
            apply_occlusion=True
        )
        
        if occlusion_loader is None:
            print("错误: 无法创建带遮挡的数据加载器，回退到标准TENT")
            use_occlusion_loss = False
        else:
            # 使用带有遮挡的数据加载器进行适应
            for step in range(steps):
                total_loss = 0
                batch_count = 0
                
                for batch in tqdm(occlusion_loader, desc=f"TENT适应 步骤 {step+1}/{steps}"):
                    optimizer.zero_grad()
                    
                    # 获取原始图像和遮挡图像
                    images = batch['image'].to(device)
                    images_occluded = batch['image_occluded'].to(device)
                    
                    # 获取嵌入
                    img_embeds = model.get_image_embeddings(images)
                    img_embeds_occluded = model.get_image_embeddings(images_occluded)
                    
                    # 计算遮挡不变性对比损失，可选择是否使用基于熵的对比损失
                    loss = occlusion_invariance_loss(
                        img_embeds, 
                        img_embeds_occluded, 
                        temperature=TEMPERATURE,
                        use_entropy=use_entropy_loss,
                        entropy_weight=ENTROPY_WEIGHT
                    )
                    
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                avg_loss = total_loss / batch_count if batch_count > 0 else 0
                print(f"步骤 {step+1}/{steps}, 平均损失: {avg_loss:.6f}")
    
    # 如果不使用遮挡不变性对比损失或创建带遮挡的数据加载器失败
    if not use_occlusion_loss:
        # 使用原始数据加载器和熵最小化进行适应
        for step in range(steps):
            total_loss = 0
            batch_count = 0
            
            for batch in tqdm(test_loader, desc=f"TENT适应 步骤 {step+1}/{steps}"):
                optimizer.zero_grad()
                
                # 获取图像
                images = batch['image'].to(device)
                
                # 获取图像嵌入
                img_embeds = model.get_image_embeddings(images)
                
                # 计算熵损失（需要将嵌入转换为logits）
                # 这里我们使用一个简单的线性层将嵌入转换为logits
                logits = torch.matmul(img_embeds, img_embeds.t())  # 使用自相似度作为logits
                loss = entropy_loss(logits)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"步骤 {step+1}/{steps}, 平均损失: {avg_loss:.6f}")
    
    # 设置回评估模式
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device, output_dir=None):
    """
    评估模型在遮挡数据集上的性能
    
    参数:
        model: 预训练模型
        test_loader: 测试数据加载器
        device: 设备
        output_dir: 输出目录（可选）
    
    返回:
        metrics: 评估指标字典
        img_embeds: 图像嵌入
        text_embeds: 文本嵌入
        all_labels: 类别标签
    """
    model.eval()
    all_img_embeds = []
    all_text_embeds = []
    all_labels = []
    all_image_paths = []
    all_texts = []
    
    # 推理阶段
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估遮挡数据集"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 获取嵌入
            img_embeds = model.get_image_embeddings(images)
            text_embeds = model.get_text_embeddings(input_ids, attention_mask)
            
            # 收集结果
            all_img_embeds.append(img_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
            all_labels.extend(batch['category'])
            all_image_paths.extend(batch['image_path'])
            all_texts.extend(batch['caption'])
    
    # 合并结果
    img_embeds = torch.cat(all_img_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # 计算指标 - 使用utils.py中的函数
    metrics = compute_metrics(img_embeds, text_embeds, all_labels)
    
    # 保存结果（可选）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_results(output_dir, metrics, all_image_paths, all_texts, img_embeds, text_embeds)
    
    return metrics, img_embeds, text_embeds, all_labels

def save_results(output_dir, metrics, image_paths, texts, img_embeds, text_embeds):
    """
    保存评估结果和示例
    
    参数:
        output_dir: 输出目录
        metrics: 评估指标
        image_paths: 图像路径列表
        texts: 文本列表
        img_embeds: 图像嵌入
        text_embeds: 文本嵌入
    """
    results = {
        "metrics": metrics,
        "examples": []
    }
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(img_embeds, text_embeds.t())
    
    # 保存前100个样本的检索结果示例
    for i in range(min(100, len(image_paths))):
        _, top5_indices = torch.topk(sim_matrix[i], 5)
        results["examples"].append({
            "image_path": image_paths[i],
            "true_text": texts[i],
            "top5_matches": [texts[idx] for idx in top5_indices.tolist()],
            "similarities": sim_matrix[i][top5_indices].tolist()
        })
    
    # 写入JSON文件
    with open(os.path.join(output_dir, "occlusion_eval_results.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评估结果已保存至 {os.path.join(output_dir, 'occlusion_eval_results.json')}")

def main():
    # 使用硬编码的参数
    model_path = MODEL_PATH
    dataset_path = DATASET_PATH
    metadata_path = METADATA_PATH
    output_dir = OUTPUT_DIR
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS
    device = DEVICE
    tent_steps = TENT_STEPS
    tent_lr = TENT_LR
    use_occlusion_loss = USE_OCCLUSION_LOSS
    use_entropy_loss = USE_ENTROPY_LOSS
    
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    print(f"数据集路径: {dataset_path}")
    print(f"元数据路径: {metadata_path}")
    print(f"输出目录: {output_dir}")
    print(f"TENT步数: {tent_steps}")
    print(f"TENT学习率: {tent_lr}")
    print(f"使用遮挡不变性损失: {use_occlusion_loss}")
    print(f"使用基于熵的对比损失: {use_entropy_loss}")
    
    # 加载预训练权重
    print(f"加载预训练模型: {model_path}")
    try:
        # 尝试加载checkpoint格式
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从checkpoint中提取模型参数
        if 'args' in checkpoint:
            model_args = checkpoint['args']
            embed_dim = model_args.get('embed_dim', 512)
            image_model = model_args.get('image_model', 'resnet50')
            text_model = model_args.get('text_model', 'roberta')
            print(f"从模型中提取参数: {image_model} + {text_model}, 嵌入维度: {embed_dim}")
        else:
            # 如果checkpoint中没有args，使用默认值
            embed_dim = 512
            image_model = 'resnet50'
            text_model = 'roberta'
            print(f"使用默认参数: {image_model} + {text_model}, 嵌入维度: {embed_dim}")
        
        # 创建模型
        model = create_model(
            embed_dim=embed_dim,
            image_model_type=image_model,
            text_model_type=text_model
        )
        
        # 加载模型权重
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 尝试直接加载state_dict
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("尝试使用默认参数创建模型...")
        
        # 使用默认参数创建模型
        model = create_model(embed_dim=512, image_model_type='resnet50', text_model_type='roberta')
        try:
            # 尝试直接加载state_dict
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"无法加载模型权重: {e}")
            return
    
    model.to(device)
    
    # 创建遮挡数据集的DataLoader
    print(f"加载遮挡数据集: {dataset_path}")
    print(f"使用元数据: {metadata_path}")
    test_loader = create_occlusion_dataloader(
        data_dir=dataset_path,
        metadata_path=metadata_path,
        text_model_type=text_model,
        image_model_type=image_model,
        batch_size=batch_size,
        num_workers=num_workers,
        apply_occlusion=False  # 评估时不需要额外遮挡
    )
    
    if test_loader is None:
        print("错误: 无法创建数据加载器")
        return
    
    # 首先评估原始模型
    print("\n评估原始模型...")
    original_metrics, orig_img_embeds, orig_text_embeds, orig_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=os.path.join(output_dir, "original")
    )
    
    print("\n原始模型评估指标:")
    for k, v in original_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 应用TENT测试时适应
    print("\n应用TENT测试时适应...")
    adapted_model = apply_tent(
        model=model,
        test_loader=test_loader,
        device=device,
        steps=tent_steps,
        lr=tent_lr,
        use_occlusion_loss=use_occlusion_loss,
        use_entropy_loss=use_entropy_loss,
        text_model_type=text_model,
        image_model_type=image_model
    )
    
    # 评估适应后的模型
    print("\n评估适应后的模型...")
    adapted_metrics, adapted_img_embeds, adapted_text_embeds, adapted_labels = evaluate_model(
        model=adapted_model,
        test_loader=test_loader,
        device=device,
        output_dir=os.path.join(output_dir, "tent_adapted")
    )
    
    print("\nTENT适应后的评估指标:")
    for k, v in adapted_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 比较原始模型和适应后的模型
    print("\n性能比较 (TENT适应 vs 原始):")
    for k in original_metrics.keys():
        diff = adapted_metrics[k] - original_metrics[k]
        print(f"{k}: {original_metrics[k]:.4f} -> {adapted_metrics[k]:.4f} (差异: {diff:+.4f})")
    
    # 生成性能对比图
    print("\n生成性能对比图...")
    metrics_keys = list(original_metrics.keys())
    metrics_values_original = [original_metrics[k] for k in metrics_keys]
    metrics_values_adapted = [adapted_metrics[k] for k in metrics_keys]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics_keys))
    width = 0.35
    
    plt.bar(x - width/2, metrics_values_original, width, label='原始模型', color='#3498db', edgecolor='black', linewidth=1.5)
    plt.bar(x + width/2, metrics_values_adapted, width, label='TENT适应后', color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    plt.xlabel('评估指标', fontsize=14)
    plt.ylabel('分数', fontsize=14)
    plt.title('原始模型 vs TENT适应后的性能对比', fontsize=16)
    plt.xticks(x, metrics_keys, fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(metrics_values_original):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    for i, v in enumerate(metrics_values_adapted):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    # 保存性能对比图
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"性能对比图已保存至: {comparison_path}")

if __name__ == "__main__":
    main()
