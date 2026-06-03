import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import TrafficSignDataset  # 假设您已有自定义Dataset类


def evaluate_model(model, test_loader, device, output_dir=None):
    """
    基础评估函数（无TTA）

    参数:
        model: 预训练模型
        test_loader: 测试集DataLoader
        device: 设备 (cuda/cpu)
        output_dir: 结果保存路径（可选）

    返回:
        metrics: 评估指标字典
    """
    model.eval()
    all_img_embeds = []
    all_text_embeds = []
    all_labels = []
    all_image_paths = []
    all_texts = []

    # 推理阶段
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
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

    # 计算指标
    metrics = compute_metrics(img_embeds, text_embeds, all_labels)

    # 保存结果（可选）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_results(output_dir, metrics, all_image_paths, all_texts, img_embeds, text_embeds)

    return metrics


def compute_metrics(img_embeds, text_embeds, labels, k_list=[1, 5, 10]):
    """
    计算图文匹配指标（与您utils.py中的函数兼容）
    """
    # 相似度矩阵 [N, N]
    sim_matrix = torch.matmul(img_embeds, text_embeds.t())
    n = len(labels)

    # 生成标签矩阵（相同类别为1）
    label_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                label_matrix[i, j] = 1

    metrics = {}

    # 计算Recall@K
    for k in k_list:
        _, topk_indices = torch.topk(sim_matrix, k, dim=1)
        correct = 0
        for i in range(n):
            if label_matrix[i, topk_indices[i]].sum() > 0:
                correct += 1
        metrics[f"Recall@{k}"] = correct / n

    # 计算mAP
    ap_scores = []
    for i in range(n):
        ap = average_precision_score(label_matrix[i], sim_matrix[i])
        ap_scores.append(ap)
    metrics["mAP"] = torch.tensor(ap_scores).mean().item()

    return metrics


def save_results(output_dir, metrics, image_paths, texts, img_embeds, text_embeds):
    """保存评估结果和示例"""
    results = {
        "metrics": metrics,
        "examples": []
    }

    # 保存前100个样本的检索结果示例
    sim_matrix = torch.matmul(img_embeds, text_embeds.t())
    for i in range(min(100, len(image_paths))):
        _, top5_indices = torch.topk(sim_matrix[i], 5)
        results["examples"].append({
            "image_path": image_paths[i],
            "true_text": texts[i],
            "top5_matches": [texts[idx] for idx in top5_indices],
            "similarities": sim_matrix[i][top5_indices].tolist()
        })

    # 写入JSON文件
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


# 使用示例
if __name__ == "__main__":
    from model_architecture import create_model
    from dataset import create_dataloaders

    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(embed_dim=512, image_model_type="resnet50", text_model_type="roberta")
    model.load_state_dict(torch.load("checkpoints/model_best.pth.tar"))
    model.to(device)

    # 2. 准备测试集
    _, _, test_loader = create_dataloaders(
        data_dir="data/val_images",
        batch_size=32,
        num_workers=4
    )

    # 3. 运行评估
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir="results/eval_output"  # 结果保存路径
    )

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")