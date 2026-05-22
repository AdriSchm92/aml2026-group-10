# CNN-Transformer Hybrid Results

**Role in project:** Main proposed model — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#proposed-architecture-cnn-transformer-hybrid)

Architecture: CNN front-end (truncated ResNet-18) + 2D learnable positional embedding +
`[CLS]` token + Transformer encoder + MLP head.  
Input `(1, 128, 313)` log-mel spectrogram, output `(K,)` per-class logits.

---

## Architecture Details

| Component | Implementation | Notes |
|---|---|---|
| CNN front-end | Truncated ResNet-18 (`timm`, `features_only=True`, `out_indices=(2,)` by default) | `num_cnn_blocks=3` → stride 8, `(B, 128, 16, 40)`, 640 tokens. Override via `cnn_backbone` kwarg (e.g. `"efficientnet_b2"`) |
| Channel projection | `Conv2d(C_cnn, d_model, 1)` + `BatchNorm2d` | Decouples transformer width from backbone width |
| 2D pos embedding | `nn.Parameter(zeros(1, d_model, H', W'))`, `trunc_normal_(std=0.02)` | Preserves time×frequency spatial axes; one vector per grid cell |
| `[CLS]` token | `nn.Parameter(zeros(1, 1, d_model))`, `trunc_normal_(std=0.02)` | Global representation extracted from position 0 of encoder output |
| Transformer encoder | `nn.TransformerEncoder`, pre-norm (`norm_first=True`), GELU, `batch_first=True` | Pre-norm is more stable for from-scratch training than post-norm |
| MLP head | `Linear(d_model, d_model) → GELU → Dropout → Linear(d_model, K)` | Applied to `[CLS]` output only |
| Loss | `BCEWithLogitsLoss` + label smoothing 0.1 | Sigmoid + binary CE per class; label smoothing for Transformer training |
| Optimiser | AdamW | Warmup + cosine decay (5 warmup epochs recommended) |

---

## Default Hyperparameters

| HP | Default | Notes |
|---|---|---|
| `cnn_backbone` | `"resnet18"` | timm model name for CNN front-end. `"efficientnet_b2"` recommended for the final model. |
| `pretrained_cnn` | `False` | ImageNet pretraining for CNN backbone. Set `True` for the final model; auto-skips warm-start from `cnn_baseline.pt`. |
| `num_cnn_blocks` | 3 | Feature stage index; 3 → stride 8, 4 → stride 16. Exact token count depends on backbone. |
| `d_model` | 256 | Transformer / projection width |
| `n_heads` | 8 | Attention heads (`d_model` must be divisible) |
| `n_layers` | 4 | Transformer encoder depth |
| `dropout` | 0.1 | Applied in transformer layers and MLP head |
| `lr` | 3e-4 | AdamW; pass explicitly — train.py default is 1e-3 |
| `weight_decay` | 0.05 | AdamW; stronger regularisation standard for ViT-style models |
| `warmup_epochs` | 5 | Linear warmup before cosine decay |
| `label_smoothing` | 0.1 | Soft binary targets; standard for Transformer from-scratch training |

---

## HP Search

HP search runs on the reduced K=69 subset (`min_recordings ≥ 200`, ~26% of full data)
for 3 epochs per trial to cheaply compare structural choices.

```bash
python scripts/hp_search.py --model cnn_transformer \
    --n_trials 6 --max_hours 8 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs
```

HP grid: `configs/hp_cnn_transformer.yaml` — covers `num_cnn_blocks`, `d_model`, `n_heads`,
`n_layers`, `dropout`, `lr`, `weight_decay`.

---

## Training

### HP search + final retrain (from-scratch ResNet-18 backbone)

```bash
# Step 1: HP search on K=69 subset (~6–8h)
python scripts/hp_search.py --model cnn_transformer \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs

# Step 2: Final retrain on full data with best config
python train.py --model cnn_transformer \
    --model_kwargs '{"d_model": 256, "n_layers": 4, "n_heads": 8, "num_cnn_blocks": 3}' \
    --epochs 15 \
    --lr 3e-4 --weight_decay 0.05 \
    --warmup_epochs 5 --label_smoothing 0.1 \
    --batch_size 256 --num_workers 10 \
    --compile \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs \
    --tag final
```

### Final model (EfficientNet-B2, pretrained CNN backbone)

Uses ImageNet-pretrained EfficientNet-B2 as the CNN front-end. Warm-start from
`best_cnn_baseline.pt` is automatically skipped when `pretrained_cnn=true`.
Use lower LR (3e-5–1e-4) since pretrained weights need gentler fine-tuning.

```bash
python train.py --model cnn_transformer \
    --model_kwargs '{"cnn_backbone": "efficientnet_b2", "pretrained_cnn": true, "d_model": 256, "n_layers": 4, "n_heads": 8}' \
    --epochs 15 \
    --lr 1e-4 --weight_decay 0.05 \
    --warmup_epochs 5 --label_smoothing 0.1 \
    --batch_size 128 --num_workers 10 \
    --compile \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs \
    --tag final_pretrained
```

### CNN warm-start (default-on)

If `best_cnn_baseline.pt` exists in the output dir, the CNN front-end is automatically
initialised from it. Disable with `--no_init`.

### Smoke test

```bash
python train.py --model cnn_transformer \
    --model_kwargs '{"d_model": 64, "n_layers": 1, "n_heads": 4}' \
    --epochs 1 --batch_size 8 --limit_train_batches 4 --limit_val_batches 4
```

### GPU utilization tips

The default `--batch_size 64` uses only ~20% of an 8 GB GPU. Scale up for better utilization:

| GPU VRAM | Recommended `--batch_size` | LR scaling vs. default (`lr * bs / 64`) |
|---|---|---|
| 8 GB | 256 | 4× base LR |
| 16 GB | 512 | 8× base LR |
| 24 GB+ | 512–1024 | 8–16× base LR |

Add `--compile` for ~20–40% throughput gain (PyTorch 2+, CUDA only; first epoch ~30s slower due to compilation warmup).

LR linear scaling rule: when changing batch size from the 64 baseline, multiply `--lr` by `new_batch_size / 64`. Combined with `--warmup_epochs 5`, this keeps training stable.

---

## Evaluation

```bash
# Val (exploratory)
python evaluate.py --model cnn_transformer

# Test (final reporting — val-tuned thresholds, no leakage)
python evaluate.py --model cnn_transformer --split test
```

---

## Results

### HP search (K=69 subset, 3 epochs/trial)

| Trial | `num_cnn_blocks` | `d_model` | `n_layers` | `n_heads` | `dropout` | `lr` | val_AUC |
|---|---|---|---|---|---|---|---|
| **0** | **4** | **256** | **4** | **4** | **0.2** | **3e-4** | **0.9057** |
| 1     | 4     | 256     | 2     | 4     | 0.2     | 1e-4   | 0.8798  |
| 2     | 4     | 128     | 2     | 8     | 0.1     | 1e-3   | 0.8614  |
| 3     | 4     | 128     | 4     | 8     | 0.2     | 1e-3   | 0.6081  |
| 4     | 4     | 256     | 2     | 8     | 0.1     | 1e-4   | 0.9040  |
| 5     | 3     | 256     | 2     | 8     | 0.2     | 1e-3   | 0.6909  |

**Best config:** Trial 0 — `num_cnn_blocks=4, d_model=256, n_layers=4, n_heads=4, dropout=0.2, lr=3e-4`

Key observations: trials with `lr=1e-3` all collapsed (AUC 0.61–0.86) — too high for Transformer training. `num_cnn_blocks=4` (stride-16, ~160 tokens) consistently outperforms `num_cnn_blocks=3` (stride-8, ~640 tokens) at equal LR.

### Final retrain (K=206, full data, correct HPs)

Config: `num_cnn_blocks=4, d_model=256, n_layers=4, n_heads=4, dropout=0.2, lr=6e-4` (Trial 0 best HP; LR linearly scaled from 3e-4@batch64 → 6e-4@batch128).  
20 epochs, 5-epoch warmup → cosine decay, label_smoothing=0.1, batch=128, spectrogram cache warm (~329s/epoch).

| Epoch | train_loss | val_AUC    | val_F1     | time (s) | LR       |
|---|---|---|---|---|---|
| 1     | 0.22399    | 0.5462     | 0.0088     | 371      | 2.400e-4 |
| 2     | 0.21193    | 0.5566     | 0.0135     | 328      | 3.600e-4 |
| 3     | 0.21153    | 0.5899     | 0.0146     | 328      | 4.800e-4 |
| 4     | 0.21124    | 0.6570     | 0.0166     | 328      | 6.000e-4 |
| 5     | 0.21049    | 0.7949     | 0.0709     | 329      | 6.000e-4 |
| 6     | 0.20949    | 0.8332     | 0.1168     | 329      | 5.934e-4 |
| 7     | 0.20861    | 0.8505     | 0.1944     | 330      | 5.741e-4 |
| 8     | 0.20780    | 0.8829     | 0.2707     | 329      | 5.427e-4 |
| 9     | 0.20710    | 0.8939     | 0.3062     | 330      | 5.007e-4 |
| 10    | 0.20648    | 0.8947     | 0.3427     | 329      | 4.500e-4 |
| 11    | 0.20596    | 0.8953     | 0.3257     | 329      | 3.927e-4 |
| 12    | 0.20552    | 0.8909     | 0.3787     | 329      | 3.314e-4 |
| 13    | 0.20513    | 0.9105     | 0.3910     | 330      | 2.686e-4 |
| 14    | 0.20479    | 0.8828     | 0.3689     | 330      | 2.073e-4 |
| 15    | 0.20447    | 0.9157     | 0.4288     | 330      | 1.500e-4 |
| 16    | 0.20418    | 0.9126     | 0.4361     | 330      | 9.926e-5 |
| 17    | 0.20396    | 0.9194     | 0.4368     | 330      | 5.729e-5 |
| 18    | 0.20377    | 0.9194     | 0.4387     | 329      | 2.594e-5 |
| 19    | 0.20364    | 0.9169     | 0.4403     | 330      | 6.556e-6 |
| **20**| **0.20359**| **0.9206** | **0.4417** | 329      | 0.000e+0 |

Best checkpoint: epoch 20 — **val_AUC 0.9206, val_F1 0.4417**

### Comparison (70/15/15 split, seed=42)

| Model | val_AUC | val_F1 | test_AUC | test_F1 | Params | Notes |
|---|---|---|---|---|---|---|
| `rf_baseline` (MFCC + RF) | 0.7653 | 0.1433 | — | — | — | 8/12 HP combos; no DL |
| `cnn_baseline` (ResNet-18) | 0.9540 | 0.4844 | — | — | ~11M | test eval pending |
| `vit_baseline` (ViT-Small, scratch) | 0.8922 | 0.3363 | — | — | ~22M | test eval pending |
| `cnn_transformer` (this) | 0.9206 | 0.4417 | — | — | ~18M | correct HP: num_cnn_blocks=4, n_heads=4 |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9537 | 0.5753 | — | — | ~22M | test eval pending |

All test_AUC/test_F1 cells pending — test evaluation not yet run.
