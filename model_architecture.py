"""Reference image-text dual encoder used by the evaluation scripts.

The original experiments use a model factory named ``create_model``. This file
provides a compatible implementation for repository users. If a released
checkpoint was trained with a different internal module naming scheme, adjust
this file or load the checkpoint with ``strict=False``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel


TEXT_MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
}


def _build_image_encoder(image_model_type: str) -> tuple[nn.Module, int]:
    name = image_model_type.lower()
    if name in {"resnet50", "resnet"}:
        model = models.resnet50(weights=None)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, out_dim
    if name in {"resnext50", "resnext50_32x4d", "resnext"}:
        model = models.resnext50_32x4d(weights=None)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, out_dim
    if name in {"efficientnet_b0", "efficientnet"}:
        model = models.efficientnet_b0(weights=None)
        out_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, out_dim
    raise ValueError(f"Unsupported image_model_type: {image_model_type}")


def _text_model_name(text_model_type: str) -> str:
    key = text_model_type.lower()
    return TEXT_MODEL_NAMES.get(key, text_model_type)


class ImageTextDualEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        image_model_type: str = "resnet50",
        text_model_type: str = "roberta",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_model_type = image_model_type
        self.text_model_type = text_model_type

        self.image_encoder, image_dim = _build_image_encoder(image_model_type)
        self.text_encoder = AutoModel.from_pretrained(_text_model_name(text_model_type))
        text_dim = self.text_encoder.config.hidden_size

        self.image_projection = nn.Linear(image_dim, embed_dim)
        self.text_projection = nn.Linear(text_dim, embed_dim)

    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return F.normalize(self.image_projection(features), dim=-1)

    def get_text_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return F.normalize(self.text_projection(features), dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_embeds = self.get_image_embeddings(images)
        text_embeds = self.get_text_embeddings(input_ids, attention_mask)
        logits = image_embeds @ text_embeds.t()
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits": logits,
        }


def create_model(
    embed_dim: int = 512,
    image_model_type: str = "resnet50",
    text_model_type: str = "roberta",
    **_: object,
) -> ImageTextDualEncoder:
    return ImageTextDualEncoder(
        embed_dim=embed_dim,
        image_model_type=image_model_type,
        text_model_type=text_model_type,
    )
