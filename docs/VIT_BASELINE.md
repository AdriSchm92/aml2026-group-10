# ViT Baseline Results (ViT-Small/16)

**Role in project:** "ViT baseline" — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#baselines)

Architecture: ViT-Small, patch 16×16, `img_size=(128, 320)`, `in_chans=1`, trained from scratch.  
Input (1, 128, 313) zero-padded to (1, 128, 320) → 8×20 = **160 tokens** + [CLS].  
Loss: BCEWithLogitsLoss (label_smoothing=0.1). Optimizer: AdamW (lr=3e-4, wd=0.05).  
Scheduler: 5-epoch linear warmup → cosine decay. SpecAugment.  
~22M params vs ResNet-18 ~11M — larger capacity needed for ViT to be competitive without CNN inductive bias.

## Training Run

### Results (70/15/15 split, seed=42)

30 epochs, 5-epoch linear warmup → cosine decay. Spectrogram cache enabled (~480s/epoch).

| Epoch | train_loss | val_AUC    | val_F1     | time (s) | LR       |
|-------|------------|------------|------------|----------|----------|
| 1     | 0.22218    | 0.5411     | 0.0096     | 569      | 1.200e-4 |
| 2     | 0.21168    | 0.6002     | 0.0128     | 479      | 1.800e-4 |
| 3     | 0.21153    | 0.5775     | 0.0096     | 479      | 2.400e-4 |
| 4     | 0.21142    | 0.6238     | 0.0155     | 480      | 3.000e-4 |
| 5     | 0.21101    | 0.7275     | 0.0411     | 479      | 3.000e-4 |
| 6     | 0.21041    | 0.8142     | 0.0877     | 480      | 2.988e-4 |
| 7     | 0.20973    | 0.8488     | 0.1552     | 481      | 2.953e-4 |
| 8     | 0.20900    | 0.8626     | 0.2063     | 480      | 2.895e-4 |
| 9     | 0.20822    | 0.8760     | 0.2505     | 481      | 2.814e-4 |
| 10    | 0.20741    | 0.8797     | 0.2843     | 481      | 2.714e-4 |
| 11    | 0.20659    | 0.8855     | 0.3084     | 482      | 2.593e-4 |
| 12    | 0.20580    | 0.8922     | 0.3363     | 483      | 2.456e-4 |
| **13**| **0.20506**| **0.8922** | **0.3479** | 483      | 2.304e-4 |
| 14    | 0.20444    | 0.8904     | 0.3592     | 483      | 2.139e-4 |
| 15    | 0.20385    | 0.8894     | 0.3624     | 483      | 1.964e-4 |
| 16    | 0.20333    | 0.8780     | 0.3726     | 483      | 1.781e-4 |
| 17    | 0.20287    | 0.8797     | 0.3651     | 482      | 1.594e-4 |
| 18    | 0.20249    | 0.8634     | 0.3801     | 483      | 1.406e-4 |
| 19    | 0.20214    | 0.8762     | 0.3848     | 483      | 1.219e-4 |
| 20    | 0.20182    | 0.8717     | 0.3811     | 483      | 1.036e-4 |
| 21    | 0.20152    | 0.8648     | 0.3832     | 483      | 8.613e-5 |
| 22    | 0.20127    | 0.8564     | 0.3823     | 483      | 6.963e-5 |
| 23    | 0.20105    | 0.8595     | 0.3937     | 483      | 5.439e-5 |
| 24    | 0.20085    | 0.8532     | 0.3880     | 484      | 4.065e-5 |
| 25    | 0.20069    | 0.8547     | 0.3870     | 484      | 2.865e-5 |
| 26    | 0.20056    | 0.8514     | 0.3869     | 483      | 1.855e-5 |
| 27    | 0.20044    | 0.8488     | 0.3959     | 483      | 1.053e-5 |
| 28    | 0.20038    | 0.8488     | 0.3965     | 483      | 4.713e-6 |
| 29    | 0.20032    | 0.8447     | 0.3951     | 483      | 1.183e-6 |
| 30    | 0.20031    | 0.8442     | 0.3939     | 483      | 0.000e+0 |

Best checkpoint: epoch 13 — **val_AUC 0.8922, val_F1 0.3479**

The model learns slowly during warmup (epochs 1–5, LR ramping to 3e-4) then improves steadily until epoch 13. AUC degrades noticeably after that — clear overfitting from epoch 16 onward as the LR cosine decays. The loss curve barely moves throughout (0.222 → 0.200), showing the model is stuck near a local optimum characteristic of training ViT from scratch without CNN inductive bias.

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

HP grid: `configs/hp_vit_baseline.yaml` — covers `drop_path_rate`, `drop_rate`, `lr`, `weight_decay`.  
6 trials × 5 epochs on K=69 subset (`min_recordings ≥ 200`).

```bash
python scripts/hp_search.py --model vit_baseline --n_trials 6 \
    --data_root $DATA_ROOT --spec_cache_dir $BIRDCLEF_SPEC_CACHE
```

### HP search results (K=69 subset, 5 epochs/trial)

| Trial | `drop_path_rate` | `drop_rate` | `lr` | `weight_decay` | val_AUC |
|---|---|---|---|---|---|
| **0** | **0.1** | **0.1** | **1e-4** | **0.05** | **0.9090** |
| 1 | 0.2 | 0.1 | 3e-4 | 0.05 | 0.8765 |
| 2 | 0.1 | 0.1 | 1e-3 | 0.01 | 0.7058 |
| 3 | 0.2 | 0.05 | 3e-4 | 0.01 | 0.8977 |
| 4 | 0.05 | 0.1 | 1e-3 | 0.01 | 0.6512 |
| 5 | 0.1 | 0.05 | 1e-3 | 0.01 | 0.6618 |

**Best config:** Trial 0 — `drop_path_rate=0.1, drop_rate=0.1, lr=1e-4, weight_decay=0.05` → val_AUC 0.9090

Key observations: `lr=1e-3` collapses training in all three trials (AUC 0.65–0.71), confirming ViT from scratch requires careful LR selection — same pattern as CNN-Transformer. `weight_decay=0.05` (stronger regularisation) consistently outperforms `0.01`. Lower LR (1e-4) with high dropout (0.1+0.1) gives the best result.

---

## Comparison vs CNN baseline

| Model | val_AUC | val_F1 | Params | Split |
|---|---|---|---|---|
| `cnn_baseline` (ResNet-18) | 0.9540 | 0.4844 | ~11M | 70/15/15 |
| `vit_baseline` (ViT-Small) | 0.8922 | 0.3363 | ~22M | 70/15/15 |

Delta: −0.062 AUC, −0.148 F1. This isolates the value of CNN inductive bias — local time-frequency pattern detection (harmonics, chirp onsets) that raw 16×16 patch projections cannot capture without a convolutional front-end.

## Next comparison models

1. `pretrained_transformer` — same ViT-Small architecture but with ImageNet pretraining — [docs/TRAINING.md](TRAINING.md)
2. `cnn_transformer` — CNN front-end + Transformer encoder (main proposed model) — [docs/CNN_TRANSFORMER.md](CNN_TRANSFORMER.md)
