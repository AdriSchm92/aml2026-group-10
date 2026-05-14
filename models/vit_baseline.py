"""ViT baseline (PROBLEMSETTING.md §Baselines — "ViT baseline").

Standard Vision Transformer on raw spectrogram patches, no CNN front-end.
Isolates the contribution of the CNN front-end vs the main CNN-Transformer model.

Input:  (B, 1, 128, 313)  — log-mel spectrogram, single channel
Padded: (B, 1, 128, 320)  — 7 zero-frames appended to time axis
Tokens: 8×20 = 160 patches of size 16×16, plus [CLS]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTBaseline(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            in_chans=1,
            img_size=(128, 320),
            num_classes=num_classes,
            pretrained=False,
            drop_path_rate=0.1,
            drop_rate=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 7))  # (B,1,128,313) → (B,1,128,320)
        return self.vit(x)


def build_model(num_classes: int, **_ignored) -> nn.Module:
    return ViTBaseline(num_classes)
