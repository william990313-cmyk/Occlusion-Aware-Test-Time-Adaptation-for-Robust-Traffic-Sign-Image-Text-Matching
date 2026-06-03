#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================
#                 Occlusion-TENT 评估 & 扫描
# ==============================================================

# ---------------- 依赖 ----------------
import os, json, random, itertools, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime

# ---------------- 自定义模块 ----------------
from model_architecture import create_model  # ← 你的模型工厂
from utils import compute_metrics  # ← 你的评估函数


# ==============================================================
#                 0. 日志配置
# ==============================================================
def setup_logger(log_dir="./logs", log_level=logging.INFO):
    """设置结构化日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"occlusion_tent_{timestamp}.log")

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除现有处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    logging.info(f"日志文件已创建: {log_file}")
    return logger


# ==============================================================
#                 1. 命令行解析（所有参数可覆写）
# ==============================================================
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Occlusion-TENT 扫描 / 单次评估")
    # ★ 路径
    p.add_argument("--model_path", type=str,
                   default="checkpoints/model_best.pth.tar")
    p.add_argument("--dataset_path", type=str,
                   default="data/occluded_images")
    p.add_argument("--metadata_path", type=str,
                   default="data/val_metadata.json")
    p.add_argument("--output_dir", type=str, default="./occlusion_eval_results")
    p.add_argument("--log_dir", type=str, default="./logs")
    # ★ DataLoader & 设备
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max_samples", type=int, default=0,
                   help="评估的最大样本数，0表示使用全部样本")
    # ★ TENT 超参
    p.add_argument("--tent_steps", type=int, default=10)
    p.add_argument("--tent_lr", type=float, default=1e-5)
    p.add_argument("--use_occ", action="store_true")
    p.add_argument("--use_ent", action="store_true")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--ent_weight", type=float, default=0.1)
    # ★ 参数扫描配置
    p.add_argument("--sweep_steps", type=str, default="1,5,10",
                   help="TENT步数扫描列表，逗号分隔")
    p.add_argument("--sweep_lr", type=str, default="5e-6,1e-5,5e-5",
                   help="学习率扫描列表，逗号分隔")
    p.add_argument("--sweep_occ", type=str, default="False,True",
                   help="是否使用遮挡损失，逗号分隔")
    p.add_argument("--sweep_ent", type=str, default="False,True",
                   help="是否使用熵损失，逗号分隔")
    # ★ 运行模式
    p.add_argument("--single", action="store_true",
                   help="只跑一次评估而不是扫描参数网格")
    p.add_argument("--resume", action="store_true",
                   help="从上次中断的地方继续参数扫描")
    return p


# ==============================================================
#                 2. 数据集 & DataLoader
# ==============================================================
class OcclusionDataset(Dataset):
    def __init__(self, data_dir, metadata_path,
                 text_model_type='roberta',  # 统一默认值为roberta
                 image_model_type='resnet50',
                 apply_occlusion=False,
                 max_samples=0):
        self.data_dir = data_dir
        self.apply_occlusion = apply_occlusion
        self.text_model_type = text_model_type
        self.image_model_type = image_model_type
        self.failed_images = []
        self.failed_count = 0

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # 如果设置了最大样本数，则限制数据集大小
        if max_samples > 0 and max_samples < len(self.metadata):
            logging.info(f"限制数据集大小为 {max_samples} 样本（原始大小: {len(self.metadata)}）")
            self.metadata = self.metadata[:max_samples]

        self.image_size = 380 if 'efficientnet' in image_model_type else 224
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if text_model_type == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif text_model_type == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 128

        logging.info(
            f"已创建数据集: {len(self.metadata)} 样本, 文本模型: {text_model_type}, 图像模型: {image_model_type}")

    def __len__(self):
        return len(self.metadata)

    # —— 随机遮挡 —— #
    def apply_random_occlusion(self, img_t):
        img_np = img_t.numpy()
        h = w = self.image_size
        oh = random.randint(h // 8, h // 4)
        ow = random.randint(w // 8, w // 4)
        ox = random.randint(0, w - ow)
        oy = random.randint(0, h - oh)
        img_np[:, oy:oy + oh, ox:ox + ow] = 0
        return torch.from_numpy(img_np)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = item['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.data_dir, os.path.basename(img_path))
        caption = item.get('captions', [''])[0] or item.get('caption', '')
        category = item['category']

        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            if self.apply_occlusion:
                img_occ = self.apply_random_occlusion(img)
        except Exception as e:
            self.failed_count += 1
            self.failed_images.append(img_path)
            logging.warning(f"加载图像失败 {img_path}: {e}")
            img = torch.zeros(3, self.image_size, self.image_size)
            img_occ = img.clone()

            # 每10个失败记录一次汇总
            if self.failed_count % 10 == 0:
                logging.warning(f"已累计 {self.failed_count} 个图像加载失败")

        enc = self.tokenizer(caption, max_length=self.max_length,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        sample = {
            'image': img,
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'category': category,
            'image_path': img_path,
            'caption': caption
        }
        if self.apply_occlusion:
            sample['image_occluded'] = img_occ
        return sample

    def get_failed_images_report(self):
        """返回失败图像的报告"""
        if not self.failed_images:
            return "没有图像加载失败"

        report = f"共有 {self.failed_count} 个图像加载失败:\n"
        for i, path in enumerate(self.failed_images[:20]):  # 只显示前20个
            report += f"{i + 1}. {path}\n"

        if len(self.failed_images) > 20:
            report += f"... 以及其他 {len(self.failed_images) - 20} 个\n"

        return report


def create_occlusion_dataloader(data_dir, metadata_path,
                                text_model_type, image_model_type,
                                batch_size=64, num_workers=4,
                                apply_occlusion=False,
                                max_samples=0):
    ds = OcclusionDataset(data_dir, metadata_path,
                          text_model_type, image_model_type,
                          apply_occlusion, max_samples)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


# ==============================================================
#                 3. TENT 辅助函数
# ==============================================================
def configure_model_for_tent(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.image_encoder.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in m.parameters():
                p.requires_grad = True

    # 记录可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        f"TENT配置: 可训练参数 {trainable_params}/{total_params} ({trainable_params / total_params * 100:.2f}%)")

    return model


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return ent.mean()


def occlusion_invariance_loss(orig, occ, temperature=0.1,
                              use_entropy=False, entropy_weight=0.1):
    orig = F.normalize(orig, p=2, dim=1)
    occ = F.normalize(occ, p=2, dim=1)
    sim = torch.matmul(orig, occ.t()) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = F.cross_entropy(sim, labels)
    if use_entropy:
        sim_probs = F.softmax(sim, dim=1)
        ent = -torch.sum(sim_probs * torch.log(sim_probs + 1e-10), dim=1).mean()
        loss = loss - entropy_weight * ent
    return loss


def apply_tent(model, loader, device, *,
               steps=10, lr=1e-5,
               use_occlusion_loss=False, use_entropy_loss=False,
               temperature=0.1, entropy_weight=0.1,
               text_model_type='roberta', image_model_type='resnet50'):
    model = configure_model_for_tent(model)
    model.train()
    optim_t = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    logging.info(
        f"开始TENT适应: steps={steps}, lr={lr}, use_occlusion={use_occlusion_loss}, use_entropy={use_entropy_loss}")
    start_time = time.time()

    if use_occlusion_loss:
        occ_loader = create_occlusion_dataloader(
            data_dir=loader.dataset.data_dir,
            metadata_path=loader.dataset.metadata_path,
            text_model_type=text_model_type,
            image_model_type=image_model_type,
            batch_size=loader.batch_size,
            num_workers=loader.num_workers,
            apply_occlusion=True,
            max_samples=0 if not hasattr(loader.dataset, 'max_samples') else loader.dataset.max_samples
        )
        for s in range(steps):
            step_losses = []
            for batch in tqdm(occ_loader, desc=f"OCC-TENT {s + 1}/{steps}"):
                optim_t.zero_grad()
                img = batch['image'].to(device)
                img_occ = batch['image_occluded'].to(device)
                emb = model.get_image_embeddings(img)
                emb_occ = model.get_image_embeddings(img_occ)
                loss = occlusion_invariance_loss(
                    emb, emb_occ, temperature,
                    use_entropy_loss, entropy_weight)
                loss.backward();
                optim_t.step()
                step_losses.append(loss.item())
            avg_loss = sum(step_losses) / len(step_losses)
            logging.info(f"OCC-TENT 步骤 {s + 1}/{steps}, 平均损失: {avg_loss:.6f}")
    else:
        for s in range(steps):
            step_losses = []
            for batch in tqdm(loader, desc=f"TENT {s + 1}/{steps}"):
                optim_t.zero_grad()
                img = batch['image'].to(device)
                emb = model.get_image_embeddings(img)
                logits = torch.matmul(emb, emb.t())
                loss = entropy_loss(logits)
                loss.backward();
                optim_t.step()
                step_losses.append(loss.item())
            avg_loss = sum(step_losses) / len(step_losses)
            logging.info(f"TENT 步骤 {s + 1}/{steps}, 平均损失: {avg_loss:.6f}")

    elapsed = time.time() - start_time
    logging.info(f"TENT适应完成，耗时: {elapsed:.2f}秒")
    model.eval()
    return model


# ==============================================================
#                 4. 评估 & 结果保存
# ==============================================================
def evaluate_model(model, loader, device, output_dir=None):
    model.eval()
    img_e, txt_e, labels, paths, caps = [], [], [], [], []

    logging.info(f"开始评估模型...")
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估"):
            img = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            img_e.append(model.get_image_embeddings(img).cpu())
            txt_e.append(model.get_text_embeddings(ids, mask).cpu())
            labels.extend(batch['category'])
            paths.extend(batch['image_path'])
            caps.extend(batch['caption'])

    img_e = torch.cat(img_e);
    txt_e = torch.cat(txt_e)
    metrics = compute_metrics(img_e, txt_e, labels)

    elapsed = time.time() - start_time
    logging.info(f"评估完成，耗时: {elapsed:.2f}秒")
    logging.info(f"评估指标: {metrics}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sim = torch.matmul(img_e, txt_e.t())
        res = {"metrics": metrics, "examples": []}
        for i in range(min(100, len(paths))):
            _, top5 = torch.topk(sim[i], 5)
            res["examples"].append({
                "image_path": paths[i],
                "true_text": caps[i],
                "top5_matches": [caps[j] for j in top5.tolist()],
                "similarities": sim[i][top5].tolist()
            })

        # 保存失败图像报告
        if hasattr(loader.dataset, 'get_failed_images_report'):
            failed_report = loader.dataset.get_failed_images_report()
            if failed_report != "没有图像加载失败":
                with open(os.path.join(output_dir, "failed_images.txt"), "w", encoding='utf-8') as f:
                    f.write(failed_report)
                logging.warning(f"已保存失败图像报告到 {output_dir}/failed_images.txt")

        # 保存结果
        result_path = os.path.join(output_dir, "results.json")
        with open(result_path, "w", encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        logging.info(f"已保存评估结果到 {result_path}")

    return metrics, img_e, txt_e, labels


# ==============================================================
#                 5. 单次评估 run_eval(cfg)
# ==============================================================
def run_eval(cfg):
    # ---------- 打印配置 ----------
    logging.info("\n======== 运行配置 ========")
    for k, v in vars(cfg).items():
        if k != "single":
            logging.info(f"{k}: {v}")
    logging.info("==========================\n")

    # ---------- 加载模型 ----------
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    try:
        ckpt = torch.load(cfg.model_path, map_location=device)
        logging.info(f"成功加载模型: {cfg.model_path}")
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise

    if 'args' in ckpt:
        embed_dim = ckpt['args'].get('embed_dim', 512)
        image_model = ckpt['args'].get('image_model', 'resnet50')
        text_model = ckpt['args'].get('text_model', 'roberta')
        logging.info(f"从检查点加载模型配置: embed_dim={embed_dim}, image_model={image_model}, text_model={text_model}")
    else:
        embed_dim, image_model, text_model = 512, 'resnet50', 'roberta'
        logging.warning(
            f"检查点中未找到模型配置，使用默认值: embed_dim={embed_dim}, image_model={image_model}, text_model={text_model}")

    model = create_model(embed_dim=embed_dim,
                         image_model_type=image_model,
                         text_model_type=text_model)

    try:
        model.load_state_dict(ckpt.get('state_dict', ckpt))
        logging.info("成功加载模型权重")
    except Exception as e:
        logging.error(f"加载模型权重失败: {e}")
        raise

    model.to(device)

    # ---------- DataLoader ----------
    loader = create_occlusion_dataloader(cfg.dataset_path, cfg.metadata_path,
                                         text_model, image_model,
                                         cfg.batch_size, cfg.num_workers,
                                         apply_occlusion=False,
                                         max_samples=cfg.max_samples)

    # ---------- 原始评估 ----------
    orig_output_dir = os.path.join(cfg.output_dir, "original")
    orig_m, _, _, _ = evaluate_model(model, loader, device, orig_output_dir)
    logging.info(f"原始模型指标: {orig_m}")

    # ---------- TENT 适应 ----------
    model_t = apply_tent(model, loader, device,
                         steps=cfg.tent_steps,
                         lr=cfg.tent_lr,
                         use_occlusion_loss=cfg.use_occ,
                         use_entropy_loss=cfg.use_ent,
                         temperature=cfg.temperature,
                         entropy_weight=cfg.ent_weight,
                         text_model_type=text_model,
                         image_model_type=image_model)

    # ---------- 适应后评估 ----------
    adapted_output_dir = os.path.join(cfg.output_dir, "tent_adapted")
    adapted_metrics, _, _, _ = evaluate_model(
        model_t, loader, device, adapted_output_dir)
    logging.info(f"TENT 适应后指标: {adapted_metrics}")

    # ---------- 对比柱状图 ----------
    keys = list(orig_m.keys())
    o_vals = [orig_m[k] for k in keys]
    a_vals = [adapted_metrics[k] for k in keys]
    x = np.arange(len(keys));
    w = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - w / 2, o_vals, w, label="Original")
    plt.bar(x + w / 2, a_vals, w, label="TENT")
    plt.xticks(x, keys, rotation=45)
    plt.ylabel("Metric Score")  # 更具体的标签
    plt.title("Performance Comparison: Original vs TENT-adapted")  # 更详细的标题
    plt.legend();
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

    os.makedirs(cfg.output_dir, exist_ok=True)
    compare_path = os.path.join(cfg.output_dir, "compare.png")
    plt.savefig(compare_path, dpi=150)
    plt.close()
    logging.info(f"已保存对比图表到 {compare_path}")

    # ---------- 返回指标字典 ----------
    return adapted_metrics


# ==============================================================
#                 6. 扫描网格 sweep_params(cfg)
# ==============================================================
def sweep_params(base_cfg):
    # 解析命令行参数中的扫描范围
    STEPS_LIST = [int(x) for x in base_cfg.sweep_steps.split(',')]
    LR_LIST = [float(x) for x in base_cfg.sweep_lr.split(',')]
    OCC_LIST = [x.lower() == 'true' for x in base_cfg.sweep_occ.split(',')]
    ENT_LIST = [x.lower() == 'true' for x in base_cfg.sweep_ent.split(',')]

    logging.info(f"参数扫描配置:")
    logging.info(f"  TENT步数: {STEPS_LIST}")
    logging.info(f"  学习率: {LR_LIST}")
    logging.info(f"  遮挡损失: {OCC_LIST}")
    logging.info(f"  熵损失: {ENT_LIST}")

    # 创建所有参数组合
    all_combinations = list(itertools.product(STEPS_LIST, LR_LIST, OCC_LIST, ENT_LIST))
    total_runs = len(all_combinations)
    logging.info(f"总共需要运行 {total_runs} 次评估")

    # 检查是否有进度文件
    progress_file = os.path.join(base_cfg.output_dir, "sweep_progress.json")
    completed_runs = []
    results = []

    if os.path.exists(progress_file) and base_cfg.resume:
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_runs = progress_data.get("completed_runs", [])
                results = progress_data.get("results", [])
            logging.info(f"从进度文件恢复，已完成 {len(completed_runs)}/{total_runs} 次评估")
        except Exception as e:
            logging.warning(f"读取进度文件失败: {e}，将从头开始")
            completed_runs = []
            results = []

    # 运行未完成的评估
    start_time = time.time()
    for i, (s, lr, occ, ent) in enumerate(all_combinations):
        # 检查是否已完成
        run_id = f"s{s}_lr{lr}_occ{occ}_ent{ent}"
        if run_id in completed_runs and base_cfg.resume:
            logging.info(f"跳过已完成的运行: {run_id}")
            continue

        logging.info(f"开始运行 {i + 1}/{total_runs}: {run_id}")
        cfg = argparse.Namespace(**vars(base_cfg))
        cfg.tent_steps, cfg.tent_lr = s, lr
        cfg.use_occ, cfg.use_ent = occ, ent
        cfg.output_dir = os.path.join(base_cfg.output_dir, run_id)

        try:
            m = run_eval(cfg)
            recall = m.get("Recall@1", 0.0)
            results.append((s, lr, occ, ent, recall))
            completed_runs.append(run_id)

            # 保存进度
            progress_data = {
                "completed_runs": completed_runs,
                "results": results,
                "last_update": datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)

            logging.info(f"完成运行 {run_id}, Recall@1: {recall:.4f}")
            logging.info(f"进度: {len(completed_runs)}/{total_runs}")
        except Exception as e:
            logging.error(f"运行 {run_id} 失败: {e}")

    elapsed = time.time() - start_time
    logging.info(f"参数扫描完成，总耗时: {elapsed:.2f}秒")

    # 绘制不同学习率下的性能曲线
    plt.figure(figsize=(12, 8))

    # 1. 无额外损失时的曲线
    plt.subplot(2, 2, 1)
    for lr in LR_LIST:
        xs, ys = zip(*[(s, r) for s, lri, occ, ent, r in results
                       if lri == lr and not occ and not ent])
        plt.plot(xs, ys, marker="o", label=f"lr={lr:g}")
    plt.xlabel("TENT Steps")
    plt.ylabel("Recall@1 Score")
    plt.title("TENT 参数扫描（无额外损失）")
    plt.grid(True)
    plt.legend()

    # 2. 使用遮挡损失时的曲线
    plt.subplot(2, 2, 2)
    for lr in LR_LIST:
        xs, ys = zip(*[(s, r) for s, lri, occ, ent, r in results
                       if lri == lr and occ and not ent])
        plt.plot(xs, ys, marker="o", label=f"lr={lr:g}")
    plt.xlabel("TENT Steps")
    plt.ylabel("Recall@1 Score")
    plt.title("TENT 参数扫描（使用遮挡损失）")
    plt.grid(True)
    plt.legend()

    # 3. 使用熵损失时的曲线
    plt.subplot(2, 2, 3)
    for lr in LR_LIST:
        xs, ys = zip(*[(s, r) for s, lri, occ, ent, r in results
                       if lri == lr and not occ and ent])
        plt.plot(xs, ys, marker="o", label=f"lr={lr:g}")
    plt.xlabel("TENT Steps")
    plt.ylabel("Recall@1 Score")
    plt.title("TENT 参数扫描（使用熵损失）")
    plt.grid(True)
    plt.legend()

    # 4. 同时使用两种损失时的曲线
    plt.subplot(2, 2, 4)
    for lr in LR_LIST:
        xs, ys = zip(*[(s, r) for s, lri, occ, ent, r in results
                       if lri == lr and occ and ent])
        plt.plot(xs, ys, marker="o", label=f"lr={lr:g}")
    plt.xlabel("TENT Steps")
    plt.ylabel("Recall@1 Score")
    plt.title("TENT 参数扫描（使用两种损失）")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    sweep_plot_path = os.path.join(base_cfg.output_dir, "tent_sweep_full.png")
    plt.savefig(sweep_plot_path, dpi=150)
    plt.close()
    logging.info(f"已保存完整扫描结果图表到 {sweep_plot_path}")

    # 找出最佳参数组合
    best_result = max(results, key=lambda x: x[4])
    best_s, best_lr, best_occ, best_ent, best_recall = best_result

    logging.info(f"最佳参数组合:")
    logging.info(f"  TENT步数: {best_s}")
    logging.info(f"  学习率: {best_lr}")
    logging.info(f"  使用遮挡损失: {best_occ}")
    logging.info(f"  使用熵损失: {best_ent}")
    logging.info(f"  Recall@1: {best_recall:.4f}")

    # 保存最佳参数
    best_params = {
        "tent_steps": best_s,
        "tent_lr": best_lr,
        "use_occlusion_loss": best_occ,
        "use_entropy_loss": best_ent,
        "recall_at_1": best_recall
    }
    best_params_path = os.path.join(base_cfg.output_dir, "best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logging.info(f"已保存最佳参数到 {best_params_path}")


# ==============================================================
#                 7. 主入口
# ==============================================================
if __name__ == "__main__":
    cfg = get_parser().parse_args()

    # 设置日志
    logger = setup_logger(cfg.log_dir)

    try:
        if cfg.single:
            run_eval(cfg)
        else:
            sweep_params(cfg)
    except Exception as e:
        logging.error(f"程序执行失败: {e}", exc_info=True)
    finally:
        logging.info("程序执行完毕")
