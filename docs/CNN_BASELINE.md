# CNN Baseline Results (ResNet-18)

**Role in project:** "ML baseline" — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#baselines)

> **Note on split:** These results were measured on the original 85/15 train/val split
> (no held-out test set). All future model runs use the 70/15/15 train/val/test split
> introduced alongside `cnn_transformer`. Direct AUC comparisons should use numbers
> from the same split.

Architecture: ResNet-18 (`timm`, `in_chans=1`) on log-mel spectrograms `(1, 128, 313)`.  
Loss: BCEWithLogitsLoss. Optimizer: AdamW (lr=1e-3, wd=1e-4). Scheduler: CosineAnnealingLR.  
Data: full BirdCLEF 2026 train set, 85/15 stratified split, SpecAugment enabled.

## Training Run


| Epoch | train_loss  | val_AUC    | val_F1     | time (s) | LR       |
| ----- | ----------- | ---------- | ---------- | -------- | -------- |
| 1     | 0.02224     | 0.9268     | 0.3139     | 9212     | 9.045e-4 |
| 2     | 0.01243     | 0.9556     | 0.4700     | 9222     | 6.545e-4 |
| 3     | 0.00882     | 0.9564     | 0.5169     | 9233     | 3.455e-4 |
| **4** | **0.00636** | **0.9570** | **0.5331** | 9231     | 9.549e-5 |


Best checkpoint: epoch 4 — **val_AUC 0.9570, val_F1 0.5331**

AUC plateaued after epoch 2 (~0.956). F1 still improving slowly via better threshold calibration. LR cosine-exhausted after epoch 4; additional epochs unlikely to yield meaningful gain.

## Next comparison models

1. `vit_baseline` — pure ViT on raw patches, no CNN front-end — [docs/VIT_BASELINE.md](VIT_BASELINE.md)
2. `cnn_transformer` — CNN front-end + Transformer encoder (main proposed model) — [docs/CNN_TRANSFORMER.md](CNN_TRANSFORMER.md)