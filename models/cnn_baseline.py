"""ResNet baseline (PROBLEMSETTING.md §Baselines — "ML baseline").

Standard ResNet applied to log-mel spectrograms with a single input channel.
Isolates the contribution of the Transformer component by comparing against a
pure CNN of similar capacity to the CNN front-end of the main model.

Hyperparameters exposed through build_model() kwargs:
  resnet_variant : timm model name. "resnet18" (default, ~11M params) or
                   "resnet34" (~21M params). Per PROBLEMSETTING §HP Tuning.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    num_classes: int,
    *,
    resnet_variant: str = "resnet18",
    **_ignored,
) -> nn.Module:
    return timm.create_model(
        resnet_variant,
        in_chans=1,
        num_classes=num_classes,
        pretrained=False,
    )
