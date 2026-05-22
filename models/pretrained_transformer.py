"""Pretrained Vision Transformer fine-tuned on mel spectrograms.

Used to evaluate whether ImageNet pretraining transfers to audio spectrogram
classification, and as a direct comparison against vit_baseline.py (same
architecture, pretrained=True vs pretrained=False).

timm handles two adaptations automatically when loading pretrained weights:
  - Patch embedding: RGB (3-ch) weights are averaged to single-channel.
  - Positional embedding: bicubic interpolation from 14×14 (224/16, ImageNet)
    to 8×20 (128/16, 320/16) for the spectrogram grid.

Input:  (B, 1, 128, 313)  — zero-padded to (B, 1, 128, 320) before forward
Output: (B, K)            — K per-class logits (apply sigmoid for probabilities)

Hyperparameters exposed through build_model() kwargs:
  vit_model      : timm model name. Larger models are slower but more expressive.
                   "vit_small_patch16_224" — 22M params, 160 tokens (8×20)
                   "vit_base_patch16_224"  — 86M params, 160 tokens (8×20)
                   "vit_small_patch8_224"  — 22M params, 640 tokens (16×40),
                                             more spatial resolution for spectrograms
  drop_path_rate : stochastic depth regularisation (recommended: 0.1–0.2)
  drop_rate      : dropout on the final token representation

Training recommendations (fine-tuning differs from training from scratch):
  LR              : 3e-5 – 1e-4  (10× lower than from-scratch ViT)
  weight_decay    : 0.05 – 0.1   (stronger regularisation)
  warmup_epochs   : 3 – 5        (protect pretrained weights early on)
  label_smoothing : 0.1
  batch_size      : 64 – 128
  epochs          : 10 – 20
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainedTransformer(nn.Module):
    """Pretrained ViT fine-tuned for multi-label bird species classification."""

    def __init__(
        self,
        num_classes: int,
        *,
        vit_model: str = "vit_small_patch16_224",
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(
            vit_model,
            in_chans=1,
            img_size=(128, 320), #640
            num_classes=num_classes,
            pretrained=True,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, 313) log-mel spectrogram

        Returns:
            logits: (B, K) — apply sigmoid for probabilities
        """
        x = F.pad(x, (0, 7))  # (0,7): (B, 1, 128, 313) → (B, 1, 128, 320), (0,14): (B, 1, 128, 626) → (B, 1, 128, 640)
        return self.vit(x)


def build_model(
    num_classes: int,
    *,
    vit_model: str = "vit_small_patch16_224",
    drop_path_rate: float = 0.1,
    drop_rate: float = 0.1,
    **_ignored,
) -> nn.Module:
    return PretrainedTransformer(
        num_classes,
        vit_model=vit_model,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )
