"""CNN-Transformer hybrid (PROBLEMSETTING.md §Proposed Architecture).

Architecture:
  1. CNN front-end  — configurable timm backbone extracts local time-frequency features.
  2. 1×1 projection — maps CNN channels to d_model independently of backbone width.
  3. 2D learnable positional embedding — preserves time×frequency spatial structure.
  4. [CLS] token + Transformer encoder — models global long-range dependencies.
  5. MLP head on [CLS] — two-layer GELU MLP produces per-class logits.

Input:  (B, 1, 128, 313)  — log-mel spectrogram, single channel, 5-second clip
Output: (B, K)            — K per-class logits (apply sigmoid for probabilities)

Hyperparameters exposed through build_model() kwargs and wired to hp_search:
  cnn_backbone    : any timm model name supporting features_only=True.
                    "resnet18" (default, from scratch) or "efficientnet_b2" (recommended
                    for the final model — stronger features, supports pretrained).
  pretrained_cnn  : load ImageNet weights for the CNN backbone. Default False.
                    Set True for the final model; skips warm-start from cnn_baseline.pt.
  num_cnn_blocks  : which feature stage to use as CNN output.
                    2 → stride 4, 3 → stride 8 (default), 4 → stride 16.
                    Exact spatial size depends on backbone; determined at init via a
                    dummy forward pass so no backbone-specific tables are needed.
  d_model         : transformer / projection width
  n_heads         : multi-head attention heads (must divide d_model)
  n_layers        : transformer encoder depth
  dropout         : dropout rate in transformer + MLP head
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


_VALID_CNN_BLOCKS = (2, 3, 4)


class CNNTransformer(nn.Module):
    """CNN-Transformer hybrid for multi-label bird species classification."""

    def __init__(
        self,
        num_classes: int,
        *,
        num_cnn_blocks: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        cnn_backbone: str = "resnet18",
        pretrained_cnn: bool = False,
    ) -> None:
        super().__init__()
        if num_cnn_blocks not in _VALID_CNN_BLOCKS:
            raise ValueError(
                f"num_cnn_blocks must be one of {_VALID_CNN_BLOCKS}, got {num_cnn_blocks}"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads} "
                f"(head_dim would be {d_model / n_heads:.1f})"
            )
        out_idx = num_cnn_blocks - 1

        # ── 1. CNN front-end ──────────────────────────────────────────────────
        # Configurable backbone via timm features_only API. out_indices selects
        # which stage to tap: 2=stride8 (default), 3=stride16, 4=stride32.
        self.cnn = timm.create_model(
            cnn_backbone,
            in_chans=1,
            features_only=True,
            out_indices=(out_idx,),
            pretrained=pretrained_cnn,
        )

        # Determine C, H', W' via a dummy forward — works for any backbone
        # without backbone-specific channel tables.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 313) #626 when using 10-seconds clips
            feat = self.cnn(dummy)[0]
        _, cnn_channels, H, W = feat.shape

        # ── 2. Channel projection: C_cnn → d_model ───────────────────────────
        self.proj = nn.Conv2d(cnn_channels, d_model, kernel_size=1, bias=False)
        # BatchNorm after projection stabilises training alongside SpecAugment.
        self.proj_norm = nn.BatchNorm2d(d_model)

        # ── 3. 2D learnable positional embedding ─────────────────────────────
        # One learned vector per (h, w) grid cell; preserves time×frequency axes
        # unlike 1D sinusoidal encoding. Initialised to near-zero.
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, H, W))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── 4a. [CLS] token ───────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 4b. Transformer encoder ───────────────────────────────────────────
        # Pre-norm (norm_first=True) is the standard ViT-style formulation;
        # more stable for training from scratch than post-norm.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # ── 5. MLP classification head ────────────────────────────────────────
        # Applied to [CLS] output only. Two linear layers, GELU, dropout.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Standard ViT-style weight init for projection and head layers."""
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, 313) log-mel spectrogram

        Returns:
            logits: (B, K) — apply sigmoid for probabilities
        """
        B = x.shape[0]

        # CNN front-end → (B, C_cnn, H', W')
        feat = self.cnn(x)[0]

        # Project + BN → (B, d_model, H', W')
        feat = self.proj_norm(self.proj(feat))

        # Add 2D positional embedding
        feat = feat + self.pos_embed

        # Flatten spatial dims → (B, H'*W', d_model)
        feat = feat.flatten(2).transpose(1, 2)

        # Prepend [CLS] token → (B, H'*W'+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)

        # Transformer encoder → (B, H'*W'+1, d_model)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Classify from [CLS] position → (B, K)
        return self.head(tokens[:, 0])


def build_model(
    num_classes: int,
    *,
    num_cnn_blocks: int = 3,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    dropout: float = 0.1,
    cnn_backbone: str = "resnet18",
    pretrained_cnn: bool = False,
    **_ignored,
) -> nn.Module:
    return CNNTransformer(
        num_classes,
        num_cnn_blocks=num_cnn_blocks,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        cnn_backbone=cnn_backbone,
        pretrained_cnn=pretrained_cnn,
    )
