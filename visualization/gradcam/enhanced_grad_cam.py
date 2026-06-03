#!/usr/bin/env python3
"""
增强版Grad-CAM实现，专门针对图像-文本匹配任务
支持多种可视化方法和更好的热区图质量
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, List

class EnhancedGradCAM:
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []

        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        """注册前向和反向钩子函数"""
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        # 查找目标层并注册钩子
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_module = module
                break
        
        if target_module is None:
            # 如果找不到指定层，尝试一些常见的层名
            possible_layers = [
                "image_encoder.backbone.layer4",
                "image_encoder.backbone.features",
                "image_encoder.backbone.conv_head",
                "image_encoder.backbone.blocks"
            ]
            
            for layer_name in possible_layers:
                for name, module in self.model.named_modules():
                    if layer_name in name and len(list(module.children())) == 0:
                        target_module = module
                        self.target_layer_name = name
                        print(f"使用层: {name}")
                        break
                if target_module is not None:
                    break
        
        if target_module is not None:
            self.hooks.append(target_module.register_forward_hook(forward_hook))
            self.hooks.append(target_module.register_backward_hook(backward_hook))
        else:
            raise ValueError(f"找不到目标层: {self.target_layer_name}")

    def generate_heatmap(self, input_image: torch.Tensor, 
                        text_input_ids: Optional[torch.Tensor] = None,
                        text_attention_mask: Optional[torch.Tensor] = None,
                        method: str = "embedding_sum") -> np.ndarray:
        """
        生成热区图
        
        Args:
            input_image: 输入图像张量 [1, C, H, W]
            text_input_ids: 文本输入ID（可选）
            text_attention_mask: 文本注意力掩码（可选）
            method: 生成方法 ("embedding_sum", "similarity", "max_activation")
        
        Returns:
            heatmap: 热区图 numpy数组
        """
        self.model.zero_grad()
        
        # 确保输入在正确的设备上
        input_image = input_image.to(next(self.model.parameters()).device)
        
        if method == "embedding_sum":
            # 方法1: 对图像嵌入求和并反向传播
            image_embeddings = self.model.get_image_embeddings(input_image)
            loss = image_embeddings.sum()
            
        elif method == "similarity" and text_input_ids is not None:
            # 方法2: 基于图像-文本相似度
            text_input_ids = text_input_ids.to(next(self.model.parameters()).device)
            text_attention_mask = text_attention_mask.to(next(self.model.parameters()).device)
            
            image_embeddings = self.model.get_image_embeddings(input_image)
            text_embeddings = self.model.get_text_embeddings(text_input_ids, text_attention_mask)
            
            # 计算相似度
            similarity = torch.cosine_similarity(image_embeddings, text_embeddings, dim=1)
            loss = similarity.sum()
            
        elif method == "max_activation":
            # 方法3: 基于最大激活值
            _ = self.model.get_image_embeddings(input_image)
            if self.activations is not None:
                loss = self.activations.max()
            else:
                raise ValueError("无法获取激活值")
        else:
            raise ValueError(f"不支持的方法: {method}")

        # 反向传播
        loss.backward()

        # 检查是否成功获取梯度和激活
        if self.gradients is None or self.activations is None:
            raise ValueError("无法获取梯度或激活值，请检查目标层名称")

        # 获取梯度和激活
        gradients = self.gradients.cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.cpu().numpy()[0]  # [C, H, W]

        # 计算权重（对梯度在空间维度上求平均）
        weights = np.mean(gradients, axis=(1, 2))  # [C]

        # 加权组合激活图
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # 应用ReLU
        heatmap = np.maximum(heatmap, 0)

        # 归一化到[0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def generate_guided_gradcam(self, input_image: torch.Tensor,
                               text_input_ids: Optional[torch.Tensor] = None,
                               text_attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """生成Guided Grad-CAM"""
        # 首先生成普通的Grad-CAM
        heatmap = self.generate_heatmap(input_image, text_input_ids, text_attention_mask)
        
        # 生成guided backpropagation
        input_image.requires_grad_(True)
        
        if text_input_ids is not None:
            image_embeddings = self.model.get_image_embeddings(input_image)
            text_embeddings = self.model.get_text_embeddings(text_input_ids, text_attention_mask)
            similarity = torch.cosine_similarity(image_embeddings, text_embeddings, dim=1)
            loss = similarity.sum()
        else:
            image_embeddings = self.model.get_image_embeddings(input_image)
            loss = image_embeddings.sum()
        
        loss.backward()
        
        guided_gradients = input_image.grad.cpu().numpy()[0]
        guided_gradients = np.maximum(guided_gradients, 0)  # ReLU
        
        # 组合Grad-CAM和guided gradients
        heatmap_resized = cv2.resize(heatmap, (guided_gradients.shape[2], guided_gradients.shape[1]))
        guided_gradcam = guided_gradients * heatmap_resized[np.newaxis, :, :]
        
        # 归一化
        guided_gradcam = guided_gradcam - guided_gradcam.min()
        if guided_gradcam.max() > 0:
            guided_gradcam = guided_gradcam / guided_gradcam.max()
        
        return guided_gradcam

    def cleanup(self):
        """清理钩子函数"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.cleanup()

def show_cam_on_image(img: torch.Tensor, heatmap: np.ndarray, 
                     alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    将热区图叠加到原始图像上
    
    Args:
        img: 原始图像张量 [C, H, W]
        heatmap: 热区图 [H, W]
        alpha: 热区图透明度
        colormap: OpenCV颜色映射
    
    Returns:
        叠加后的图像
    """
    # 将图像张量转换为numpy数组并反归一化
    if img.dim() == 4:
        img = img.squeeze(0)
    
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # ImageNet反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    # 调整热区图大小以匹配图像
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

    # 叠加图像
    superimposed_img = heatmap_colored * alpha + img_np * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img

def create_heatmap_grid(images: List[np.ndarray], titles: List[str], 
                       figsize: Tuple[int, int] = (15, 5)) -> np.ndarray:
    """
    创建热区图网格显示
    
    Args:
        images: 图像列表
        titles: 标题列表
        figsize: 图像大小
    
    Returns:
        网格图像
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 将matplotlib图像转换为numpy数组
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return buf

