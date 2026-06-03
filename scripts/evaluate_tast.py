import os
import json
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from model_architecture import create_model
from utils import compute_metrics

# —— 全局配置 ——
MODEL_PATH    = "checkpoints/model_best.pth.tar"
DATASET_PATH  = "data/occluded_images"
METADATA_PATH = "data/val_metadata.json"
OUTPUT_DIR    = "./occlusion_eval_results"
BATCH_SIZE    = 64
NUM_WORKERS   = 4
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"

# —— TAST 超参数 ——
TAST_STEPS       = 8     # 自适应迭代步数
TAST_LR          = 1e-3  # 学习率
TAST_MOMENTUM    = 0.999 # EMA 更新系数
TAST_TEMPERATURE = 1.0   # 软化温度系数


class OcclusionDataset(Dataset):
    """加载遮挡或原始图像的数据集"""
    def __init__(self, data_dir, metadata_path,
                 text_model_type='distilbert', image_model_type='resnet50',
                 apply_occlusion=False):
        self.data_dir = data_dir
        self.apply_occlusion = apply_occlusion
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.image_size = 380 if 'efficientnet' in image_model_type else 224
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        if 'roberta' in text_model_type:
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        elif 'bert' in text_model_type:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 128

    def __len__(self):
        return len(self.metadata)

    def apply_random_occlusion(self, image):
        arr = image.numpy()
        h, w = self.image_size, self.image_size
        oh = random.randint(h//8, h//4)
        ow = random.randint(w//8, w//4)
        y  = random.randint(0, h-oh)
        x  = random.randint(0, w-ow)
        arr[:, y:y+oh, x:x+ow] = 0
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        path = item['image_path']
        if not os.path.isabs(path):
            path = os.path.join(self.data_dir, os.path.basename(path))
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        if self.apply_occlusion:
            img = self.apply_random_occlusion(img)
        caption = item.get('captions', [''])[0] or item.get('caption', '')
        enc = self.tokenizer(caption, max_length=self.max_length,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attn_mask = enc['attention_mask'].squeeze(0)
        label     = item['category']
        return {'image': img,
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'category': label}


def create_occlusion_dataloader(data_dir, metadata_path,
                                text_model_type, image_model_type,
                                batch_size=64, num_workers=4,
                                apply_occlusion=False):
    ds = OcclusionDataset(data_dir, metadata_path,
                           text_model_type, image_model_type,
                           apply_occlusion)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


def apply_tast(student, teacher, data_loader, optimizer):
    student.train()
    for step, batch in enumerate(data_loader):
        if step >= TAST_STEPS:
            break
        imgs = batch['image'].to(DEVICE)
        with torch.no_grad():
            logits_t = teacher.get_image_embeddings(imgs)
            soft_t   = F.softmax(logits_t / TAST_TEMPERATURE, dim=1)
        logits_s = student.get_image_embeddings(imgs)
        logp_s   = F.log_softmax(logits_s / TAST_TEMPERATURE, dim=1)
        loss = F.kl_div(logp_s, soft_t, reduction='batchmean') * (TAST_TEMPERATURE**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p_t, p_s in zip(teacher.parameters(), student.parameters()):
                p_t.data.mul_(TAST_MOMENTUM).add_((1 - TAST_MOMENTUM) * p_s.data)
    return teacher


def evaluate_model(model, loader, device, output_dir=None):
    model.eval()
    all_img, all_txt, labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估模型"):
            imgs = batch['image'].to(device)
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            img_e = model.get_image_embeddings(imgs)
            txt_e = model.get_text_embeddings(ids, mask)
            all_img.append(img_e.cpu())
            all_txt.append(txt_e.cpu())
            labels.extend(batch['category'])
    img_embeds = torch.cat(all_img, dim=0)
    txt_embeds = torch.cat(all_txt, dim=0)
    metrics = compute_metrics(img_embeds, txt_embeds, labels)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return metrics, img_embeds, txt_embeds, labels


def main():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'args' in ckpt:
        args = ckpt['args']
        if isinstance(args, dict):
            embed_dim    = args.get('embed_dim', 512)
            image_model  = args.get('image_model_type', 'resnet50')
            text_model   = args.get('text_model_type', 'distilbert')
        else:
            embed_dim    = getattr(args, 'embed_dim', 512)
            image_model  = getattr(args, 'image_model_type', 'resnet50')
            text_model   = getattr(args, 'text_model_type', 'distilbert')
    else:
        embed_dim, image_model, text_model = 512, 'resnet50', 'distilbert'
    student = create_model(embed_dim=embed_dim,
                           image_model_type=image_model,
                           text_model_type=text_model)
    state = ckpt.get('state_dict', ckpt)
    student.load_state_dict(state)
    student.to(DEVICE)
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    test_loader = create_occlusion_dataloader(
        DATASET_PATH, METADATA_PATH,
        text_model, image_model,
        BATCH_SIZE, NUM_WORKERS,
        apply_occlusion=False)
    orig_metrics, _, _, _ = evaluate_model(
        student, test_loader, DEVICE,
        output_dir=os.path.join(OUTPUT_DIR, 'original'))
    optimizer = optim.Adam(student.parameters(), lr=TAST_LR)
    adapted = apply_tast(student, teacher, test_loader, optimizer)
    new_metrics, _, _, _ = evaluate_model(
        adapted, test_loader, DEVICE,
        output_dir=os.path.join(OUTPUT_DIR, 'tast_adapted'))
    # 格式化性能比较输出
    keys = list(orig_metrics.keys())
    print("性能比较 (原始 -> TAST):")
    for k in keys:
        o = orig_metrics[k]
        n = new_metrics[k]
        print(f"{k}: {o:.4f} -> {n:.4f} (差异: {n-o:+.4f})")
    # 可视化对比
    ori_vals = [orig_metrics[k] for k in keys]
    new_vals = [new_metrics[k] for k in keys]
    x = np.arange(len(keys))
    plt.figure(figsize=(10,6))
    width = 0.35
    plt.bar(x-width/2, ori_vals, width, label='原始')
    plt.bar(x+width/2, new_vals, width, label='TAST')
    plt.xticks(x, keys, rotation=45)
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_tast.png'), dpi=150)
    print(f"对比图保存在 {OUTPUT_DIR}/comparison_tast.png")

if __name__ == '__main__':
    main()
