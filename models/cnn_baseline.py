"""ResNet-18 baseline (PROBLEMSETTING.md §Baselines — "ML baseline").

Standard ResNet-18 applied to log-mel spectrograms with a single input channel.
Isolates the contribution of the Transformer component by comparing against a
pure CNN of similar capacity to the CNN front-end of the main model.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_model(num_classes: int) -> nn.Module:
    return timm.create_model(
        "resnet18",
        in_chans=1,
        num_classes=num_classes,
        pretrained=False,
    )
