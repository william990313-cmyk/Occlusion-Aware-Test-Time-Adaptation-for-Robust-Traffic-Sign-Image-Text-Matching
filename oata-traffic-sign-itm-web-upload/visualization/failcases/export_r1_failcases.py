#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export R@1 failure cases with top-5 predictions (text->image).
Keeps the original script untouched by loading it dynamically.

Usage:
    python export_r1_failcases.py \
        --base-script /mnt/data/evaluate_occlusion_tent_CE_ReRank.py

It will reuse MODEL_PATH / DATA_DIR / META_PATH / OUTPUT_DIR constants
from the base script. Results go to: {OUTPUT_DIR}/failcases_text2img/
"""
import os, json, argparse, math, shutil, importlib.util
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_module_from_path(py_path: str):
    spec = importlib.util.spec_from_file_location("tent_ce_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_contact_sheet(img_paths: List[Path], out_path: Path, thumb_size=(256,256)):
    """Create a single-row contact sheet for top-5 images."""
    if not img_paths:
        return
    thumbs = []
    for p in img_paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            # fallback to blank
            im = Image.new("RGB", thumb_size, (240,240,240))
        im = ImageOps.fit(im, thumb_size, Image.BICUBIC)
        thumbs.append(im)
    w = thumb_size[0] * len(thumbs)
    h = thumb_size[1]
    sheet = Image.new("RGB", (w, h), (255,255,255))
    for i, im in enumerate(thumbs):
        sheet.paste(im, (i*thumb_size[0], 0))
    sheet.save(out_path)

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-script", type=str, required=True,
                    help="Path to evaluate_occlusion_tent_CE_ReRank.py (unchanged).")
    ap.add_argument("--max-fail", type=int, default=None,
                    help="Optional cap on number of failures to export.")
    args = ap.parse_args()

    base = load_module_from_path(args.base_script)

    # Reuse constants and helpers from the original
    DEVICE      = getattr(base, "DEVICE")
    OUTPUT_DIR  = Path(getattr(base, "OUTPUT_DIR"))
    DATA_DIR    = Path(getattr(base, "DATA_DIR"))
    META_PATH   = Path(getattr(base, "META_PATH"))
    TOPK        = getattr(base, "TOPK")
    ce_score    = getattr(base, "ce_score")
    build_loader= getattr(base, "build_loader")
    load_backbone = getattr(base, "load_backbone")
    tent_adapt  = getattr(base, "tent_adapt")

    # We need access to image file paths and captions; the original dataset
    # returns only tensors and caption text. We'll parse META JSON directly.
    meta = json.load(open(META_PATH))
    image_files = [DATA_DIR / os.path.basename(it["image_path"]) for it in meta]
    captions    = [it.get("caption", it.get("captions", [""])[0]) for it in meta]

    # Prepare dataloader for TENT
    loader = build_loader()

    # Load & adapt model (unchanged path/weights)
    print(">>> Stage-1  TENT BN-adapt (same as base)")
    model  = load_backbone().to(DEVICE)
    model  = tent_adapt(model, loader)
    model.eval()

    # Collect dual-encoder embeddings in the same way
    print(">>> Stage-2  Dual-Encoder forward (same as base)")
    img_emb, txt_emb = [], []
    with torch.no_grad():
        for b in base.tqdm(loader, desc="DualEmb"):
            img_emb.append(model.get_image_embeddings(b["image"].to(DEVICE)).cpu())
            txt_emb.append(model.get_text_embeddings(b["input_ids"].to(DEVICE),
                                                     b["attention_mask"].to(DEVICE)).cpu())
    img_emb = torch.cat(img_emb, 0)  # (N, D)
    txt_emb = torch.cat(txt_emb, 0)  # (N, D)

    # Compute initial ranking by dual-encoder similarity
    print(">>> Stage-3  Cross-Encoder Re-Rank (same as base)")
    sim   = img_emb @ txt_emb.T
    N     = sim.size(0)
    rank0 = sim.argsort(1, descending=True)             # (N,N)

    # CE re-ranking for top-K (text->image direction)
    new_rank = torch.empty_like(rank0)
    for i in base.tqdm(range(N), desc="Cross-Encoder"):
        top_idx = rank0[i, :TOPK]
        pair1   = [captions[i]] * TOPK
        pair2   = [captions[j] for j in top_idx.tolist()]
        scores  = ce_score((pair1, pair2)).detach().cpu()
        _, rer  = scores.sort(descending=True)
        new_rank[i] = torch.cat([top_idx[rer], rank0[i, TOPK:]], 0)

    # Identify R@1 failures: first rank is NOT the ground-truth index i
    lab = torch.arange(N)
    correct_at_1 = (new_rank[:, 0] == lab)
    fail_indices = torch.where(~correct_at_1)[0].tolist()

    print(f"Total samples: {N}, R@1 failures: {len(fail_indices)} "
          f"({len(fail_indices)/max(1,N)*100:.2f}%)")

    # Limit number if user wants
    if args.max_fail is not None:
        fail_indices = fail_indices[:args.max_fail]

    # Output structure
    out_root = OUTPUT_DIR / "failcases_text2img"
    ensure_dir(out_root)

    # Write a compact JSON lines file and an index HTML with thumbnails
    manifest = []
    html_lines = [
        "<html><head><meta charset='utf-8'><title>R@1 Failcases (text→image)</title></head><body>",
        f"<h2>R@1 Failures: {len(fail_indices)} / {N} ({len(fail_indices)/max(1,N)*100:.2f}%)</h2>",
        "<ol>"
    ]

    for qidx in fail_indices:
        # Top-5 predictions after CE
        top5 = new_rank[qidx, :5].tolist()
        # Build contact sheet for quick view
        item_dir = out_root / f"q{qidx:05d}"
        ensure_dir(item_dir)
        pred_paths = [image_files[j] for j in top5]

        # copy images (ranked) into folder for this query
        copied = []
        for k, p in enumerate(pred_paths, start=1):
            dst = item_dir / f"top{k}_{p.name}"
            try:
                if dst.exists():
                    dst.unlink()
                shutil.copy2(p, dst)
                copied.append(dst)
            except Exception as e:
                # skip on copy error
                pass

        # save contact sheet
        sheet_path = item_dir / "top5_contact_sheet.jpg"
        make_contact_sheet(copied, sheet_path)

        record = {
            "query_index": qidx,
            "query_caption": captions[qidx],
            "gt_image": str(image_files[qidx]),
            "top5_indices": top5,
            "top5_images": [str(pp) for pp in pred_paths]
        }
        manifest.append(record)

        # HTML block
        html_lines.append("<li style='margin-bottom:16px;'>")
        html_lines.append(f"<div><b>Query #{qidx}</b>: {captions[qidx]}</div>")
        html_lines.append(f"<div>Ground-truth image: <code>{image_files[qidx].name}</code></div>")
        if sheet_path.exists():
            rel = os.path.relpath(sheet_path, out_root)
            html_lines.append(f"<div><img src='{rel}' style='max-width:100%;height:auto;'/></div>")
        else:
            # fallback: list images
            html_lines.append("<div>Top-5 predictions:</div><ul>")
            for k, pp in enumerate(pred_paths, start=1):
                rel = os.path.relpath(pp, out_root)
                html_lines.append(f"<li>#{k}: <code>{rel}</code></li>")
            html_lines.append("</ul>")
        html_lines.append("</li>")

    html_lines.append("</ol></body></html>")

    # Save outputs
    with open(out_root / "failcases.jsonl", "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_root / "index.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"[OK] Exported to: {out_root}")
    print(f" - Manifest: {out_root / 'failcases.jsonl'}")
    print(f" - HTML index: {out_root / 'index.html'}")

if __name__ == "__main__":
    main()
