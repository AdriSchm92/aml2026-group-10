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

_Results pending — run in progress._

### HP search (K=69 subset, 3 epochs/trial)

| Trial | `num_cnn_blocks` | `d_model` | `n_layers` | `n_heads` | `dropout` | `lr` | val_AUC |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Best config:** _pending_

### Final retrain (K=206, full data)

| Epoch | train_loss | val_AUC | val_F1 | time (s) | LR |
|---|---|---|---|---|---|
| | | | | | |

**Best checkpoint:** `best_cnn_transformer_final.pt`

### Comparison (70/15/15 split, seed=42)

| Model | val_AUC | val_F1 | test_AUC | test_F1 | Params | Split |
|---|---|---|---|---|---|---|
| `rf_baseline` (MFCC + RF) | _pending_ | _pending_ | _pending_ | _pending_ | — | 70/15/15 |
| `cnn_baseline` (ResNet-18) | 0.9570* | 0.5331* | — | — | ~11M | 85/15 (old) |
| `vit_baseline` (ViT-Small) | _pending_ | _pending_ | _pending_ | _pending_ | ~22M | 70/15/15 |
| `cnn_transformer` (this) | _pending_ | _pending_ | _pending_ | _pending_ | ~15–20M | 70/15/15 |

\* CNN baseline numbers from old 85/15 split — see [CNN_BASELINE.md](CNN_BASELINE.md) caveat.
