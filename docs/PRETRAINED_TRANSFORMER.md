# Pretrained Transformer Results (ViT-Small, ImageNet)

**Role in project:** Supplementary comparison — ImageNet-pretrained ViT fine-tuned on mel spectrograms. Not part of the three formal baselines in PROBLEMSETTING.md but included to quantify the benefit of transfer learning vs. training from scratch.

Architecture: ViT-Small/16, `img_size=(128, 320)`, `in_chans=1`. ImageNet weights loaded via timm (patch embedding averaged from 3→1 channel; positional embedding bicubic-interpolated from 14×14 to 8×20). Input zero-padded from (1, 128, 313) to (1, 128, 320).  
Loss: BCEWithLogitsLoss (label_smoothing=0.1). Optimizer: AdamW (lr=1e-4, wd=0.05).  
Scheduler: 5-epoch linear warmup → cosine decay. SpecAugment enabled.  
~22M params (same architecture as `vit_baseline`, pretrained=True vs pretrained=False).

---

## Training Run

### Results (70/15/15 split, seed=42)

15 epochs, 5-epoch linear warmup → cosine decay. Spectrogram cache enabled (~484s/epoch).

| Epoch | train_loss | val_AUC    | val_F1     | time (s) | LR       |
|-------|------------|------------|------------|----------|----------|
| 1     | 0.22701    | 0.5731     | 0.0096     | 575      | 4.000e-5 |
| 2     | 0.21234    | 0.7598     | 0.0692     | 485      | 6.000e-5 |
| 3     | 0.20989    | 0.8598     | 0.2646     | 485      | 8.000e-5 |
| 4     | 0.20774    | 0.9061     | 0.3821     | 484      | 1.000e-4 |
| 5     | 0.20640    | 0.9289     | 0.4501     | 484      | 1.000e-4 |
| 6     | 0.20547    | 0.9390     | 0.4794     | 484      | 9.755e-5 |
| 7     | 0.20480    | 0.9410     | 0.5099     | 484      | 9.045e-5 |
| 8     | 0.20427    | 0.9472     | 0.5273     | 484      | 7.939e-5 |
| 9     | 0.20383    | 0.9463     | 0.5382     | 485      | 6.545e-5 |
| 10    | 0.20343    | 0.9480     | 0.5462     | 485      | 5.000e-5 |
| 11    | 0.20308    | 0.9524     | 0.5557     | 484      | 3.455e-5 |
| 12    | 0.20276    | 0.9507     | 0.5636     | 485      | 2.061e-5 |
| 13    | 0.20252    | 0.9508     | 0.5690     | 485      | 9.549e-6 |
| 14    | 0.20235    | 0.9523     | 0.5750     | 485      | 2.447e-6 |
| **15**| **0.20224**| **0.9537** | **0.5753** | 484      | 0.000e+0 |

Best checkpoint: epoch 15 — **val_AUC 0.9537, val_F1 0.5753**

The model converges dramatically faster than `vit_baseline` (same architecture, no pretraining): AUC 0.9061 by epoch 4 vs epoch 12 for `vit_baseline`. Transfer learning from ImageNet effectively provides the local feature detectors that the from-scratch ViT had to learn slowly. AUC continues improving through epoch 15 with no sign of overfitting, suggesting more epochs would help.

---

## Launch command

```bash
python train.py --model pretrained_transformer \
    --epochs 15 \
    --lr 1e-4 --weight_decay 0.05 \
    --warmup_epochs 5 --label_smoothing 0.1 \
    --batch_size 128 --compile \
    --data_root $DATA_ROOT --spec_cache_dir $BIRDCLEF_SPEC_CACHE
```

---

## Comparison vs vit_baseline (same architecture, different initialisation)

| Model | val_AUC | val_F1 | Best epoch | Params |
|---|---|---|---|---|
| `vit_baseline` (from scratch) | 0.8922 | 0.3363 | 12 / 30 | ~22M |
| `pretrained_transformer` (ImageNet) | 0.9537 | 0.5753 | 15 / 15 | ~22M |

Delta from pretraining alone: **+0.061 AUC, +0.239 F1**. This is the largest single gain in the comparison table and isolates the value of ImageNet transfer for audio spectrogram classification.

## Comparison across all models (70/15/15 split, seed=42)

See [CNN_TRANSFORMER.md](CNN_TRANSFORMER.md#comparison-70-15-15-split-seed-42) for the full table.
