"""Evaluation utilities for image-text retrieval."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from sklearn.metrics import average_precision_score


def _labels_to_matrix(labels: Sequence | None, n: int) -> torch.Tensor:
    if labels is None:
        return torch.eye(n, dtype=torch.float32)

    label_list = list(labels)
    if len(label_list) != n:
        raise ValueError(f"Expected {n} labels, received {len(label_list)}.")

    return torch.tensor(
        [[1.0 if label_list[i] == label_list[j] else 0.0 for j in range(n)] for i in range(n)],
        dtype=torch.float32,
    )


def compute_metrics(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    labels: Sequence | None = None,
    k_list: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K and mAP for image-to-text retrieval.

    If labels are provided, samples with the same label/category are treated as
    positives. If labels are omitted, the metric falls back to one-to-one
    image/text pairing.
    """

    if img_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("img_embeds and text_embeds must be 2D tensors.")
    if img_embeds.size(0) != text_embeds.size(0):
        raise ValueError("Image and text embedding batches must have the same length.")

    img_embeds = torch.nn.functional.normalize(img_embeds.detach().cpu(), dim=1)
    text_embeds = torch.nn.functional.normalize(text_embeds.detach().cpu(), dim=1)

    sim_matrix = img_embeds @ text_embeds.t()
    n = sim_matrix.size(0)
    label_matrix = _labels_to_matrix(labels, n)

    metrics: dict[str, float] = {}
    for k in k_list:
        k_eff = min(int(k), n)
        _, topk_indices = torch.topk(sim_matrix, k_eff, dim=1)
        hits = 0
        for i in range(n):
            if label_matrix[i, topk_indices[i]].sum().item() > 0:
                hits += 1
        metrics[f"Recall@{k}"] = hits / n

    ap_scores = []
    for i in range(n):
        y_true = label_matrix[i].numpy()
        y_score = sim_matrix[i].numpy()
        if y_true.sum() == 0:
            continue
        ap_scores.append(average_precision_score(y_true, y_score))
    metrics["mAP"] = float(sum(ap_scores) / len(ap_scores)) if ap_scores else 0.0

    return metrics
