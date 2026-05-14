# ViT Baseline Results (ViT-Small/16)

**Role in project:** "ViT baseline" — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#baselines)

Architecture: ViT-Small, patch 16×16, `img_size=(128, 320)`, `in_chans=1`, trained from scratch.  
Input (1, 128, 313) zero-padded to (1, 128, 320) → 8×20 = **160 tokens** + [CLS].  
Loss: BCEWithLogitsLoss (label_smoothing=0.1). Optimizer: AdamW (lr=3e-4, wd=0.05).  
Scheduler: 5-epoch linear warmup → cosine decay. SpecAugment + 85/15 stratified split.  
~22M params vs ResNet-18 ~11M — larger capacity needed for ViT to be competitive without CNN inductive bias.

## Training Run

_Results pending — run in progress._

| Epoch | train_loss | val_AUC | val_F1 | time (s) | LR |
|-------|-----------|---------|--------|----------|----|
| | | | | | |

## Launch command

```bash
python train.py --model vit_baseline \
    --epochs 30 \
    --batch_size 128 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --label_smoothing 0.1 \
    --num_workers 10 \
    --grad_clip 1.0 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs \
    --tag vit_baseline_v1
```

## Comparison vs CNN baseline

| Model | val_AUC | val_F1 | Params | Epochs |
|---|---|---|---|---|
| `cnn_baseline` (ResNet-18) | 0.9570 | 0.5331 | ~11M | 4 |
| `vit_baseline` (ViT-Small) | _pending_ | _pending_ | ~22M | 30 |

Delta isolates the value of CNN inductive bias (local time-frequency features) over raw patch projection.

## Next comparison models

1. `cnn_transformer` — CNN front-end + Transformer encoder (main proposed model) — [docs/CNN_TRANSFORMER.md](CNN_TRANSFORMER.md)
