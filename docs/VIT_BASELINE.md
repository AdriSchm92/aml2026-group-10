# ViT Baseline Results (ViT-Small/16)

**Role in project:** "ViT baseline" — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#baselines)

Architecture: ViT-Small, patch 16×16, `img_size=(128, 320)`, `in_chans=1`, trained from scratch.  
Input (1, 128, 313) zero-padded to (1, 128, 320) → 8×20 = **160 tokens** + [CLS].  
Loss: BCEWithLogitsLoss (label_smoothing=0.1). Optimizer: AdamW (lr=3e-4, wd=0.05).  
Scheduler: 5-epoch linear warmup → cosine decay. SpecAugment.  
~22M params vs ResNet-18 ~11M — larger capacity needed for ViT to be competitive without CNN inductive bias.

> **Note on split:** Preliminary numbers below are from the old **85/15** train/val split
> (no held-out test set). All final comparisons must use the **70/15/15** split
> (the current default). Rerun with the launch command below to get comparable numbers.

---

## Training Run

### Preliminary results (85/15 split — needs rerun on 70/15/15)

| Epoch | train_loss | val_AUC | val_F1 | time (s) | LR |
|-------|-----------|---------|--------|----------|----|
| best  | —         | 0.9059  | 0.4060 | —        | —  |

Best checkpoint (preliminary): **val_AUC 0.9059, val_F1 0.4060**

Delta vs CNN baseline (0.9570 / 0.5331, same 85/15 split): −0.051 AUC, −0.127 F1.  
This gap isolates the value of CNN inductive bias (local time-frequency features) over raw patch projection.

### Final results (70/15/15 split — pending rerun)

| Epoch | train_loss | val_AUC | val_F1 | time (s) | LR |
|-------|-----------|---------|--------|----------|----|
|       |           |         |        |          |    |

---

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

## HP search

```bash
python scripts/hp_search.py --model vit_baseline --n_trials 6 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs
```

HP grid: `configs/hp_vit_baseline.yaml` — covers `drop_path_rate`, `drop_rate`, `lr`, `weight_decay`.

---

## Comparison vs CNN baseline

| Model | val_AUC | val_F1 | Params | Split | Notes |
|---|---|---|---|---|---|
| `cnn_baseline` (ResNet-18) | 0.9570 | 0.5331 | ~11M | 85/15 | needs rerun on 70/15/15 |
| `vit_baseline` (ViT-Small) | 0.9059 | 0.4060 | ~22M | 85/15 | needs rerun on 70/15/15 |

Delta isolates the value of CNN inductive bias (local time-frequency features) over raw patch projection.

## Next comparison models

1. `pretrained_transformer` — same ViT-Small architecture but with ImageNet pretraining — [docs/TRAINING.md](TRAINING.md)
2. `cnn_transformer` — CNN front-end + Transformer encoder (main proposed model) — [docs/CNN_TRANSFORMER.md](CNN_TRANSFORMER.md)
