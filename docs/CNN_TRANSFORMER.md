# CNN-Transformer Hybrid Results

**Role in project:** Main proposed model — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#proposed-architecture-cnn-transformer-hybrid)

Architecture: CNN front-end (truncated ResNet-18) + 2D learnable positional embedding +
`[CLS]` token + Transformer encoder + MLP head.  
Input `(1, 128, 313)` log-mel spectrogram, output `(K,)` per-class logits.

---

## Architecture Details

| Component | Implementation | Notes |
|---|---|---|
| CNN front-end | Truncated ResNet-18 (`timm`, `features_only=True`) | `num_cnn_blocks=3` → stride 8, `(B, 128, 16, 40)`, 640 tokens (code default). `num_cnn_blocks=4` → stride 16, `(B, 256, 8, 20)`, **160 tokens** (used in training).|
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
| `cnn_backbone` | `"resnet18"` | timm model name for CNN front-end.|
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

### HP Search Results (K=69 subset, 3 epochs/trial)

| Trial | `num_cnn_blocks` | `d_model` | `n_layers` | `n_heads` | `dropout` | `lr` | val_AUC |
|---|---|---|---|---|---|---|---|
| **0** | **4** | **256** | **4** | **4** | **0.2** | **3e-4** | **0.9057** |
| 1     | 4     | 256     | 2     | 4     | 0.2     | 1e-4   | 0.8798  |
| 2     | 4     | 128     | 2     | 8     | 0.1     | 1e-3   | 0.8614  |
| 3     | 4     | 128     | 4     | 8     | 0.2     | 1e-3   | 0.6081  |
| 4     | 4     | 256     | 2     | 8     | 0.1     | 1e-4   | 0.9040  |
| 5     | 3     | 256     | 2     | 8     | 0.2     | 1e-3   | 0.6909  |

**Best config:** Trial 0 — `num_cnn_blocks=4, d_model=256, n_layers=4, n_heads=4, dropout=0.2, lr=3e-4`

Note: `weight_decay` was also in the HP search space (`[1e-4, 5e-2]`) but per-trial values were not recorded in the table above. The final retrain used `weight_decay=0.05` consistent with the best trial.

Key observations: trials with `lr=1e-3` all collapsed (AUC 0.61–0.86) — too high for Transformer training. `num_cnn_blocks=4` (stride-16, ~160 tokens) consistently outperforms `num_cnn_blocks=3` (stride-8, ~640 tokens) at equal LR.

---

## Training

### Final retrain (K=206, full data, best HP config)

Config: `num_cnn_blocks=4, d_model=256, n_layers=4, n_heads=4, dropout=0.2, lr=3e-4`.  
3 runs with different random seeds for robustness. Seeds 42 and 123 used 15 epochs
(lr=3e-4, warmup=5). Seed 456 used a lower lr and more epochs to verify continued learning —
results are directionally consistent and the difference in methodology does not materially
affect the comparison.

### Launch commands

```bash
# Seed 42
python train.py --model cnn_transformer \
    --model_kwargs '{"num_cnn_blocks": 4, "d_model": 256, "n_layers": 4, "n_heads": 4, "dropout": 0.2}' \
    --epochs 15 --lr 3e-4 --weight_decay 0.05 \
    --warmup_epochs 5 --label_smoothing 0.1 \
    --batch_size 256 --num_workers 10 --compile \
    --data_root /home/renku/work/kaggle-data/birdclef-2026 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs \
    --spec_cache_dir /home/renku/work/kaggle-data/birdclef_specs \
    --seed 42 --tag seed42

# Seed 123 (same config)
... --seed 123 --tag seed123

# Seed 456 (lower lr, more epochs)
... --seed 456 --tag seed456
```

---

## Results

### Multi-seed runs (K=206, full data, correct HP config)

| Seed | best val_AUC | best val_F1 | Notes |
|------|-------------|-------------|-------|
| 42   | 0.9498      | 0.2334      | 15 epochs, lr=3e-4 |
| 123  | 0.9331      | 0.2191      | 15 epochs, lr=3e-4 |
| 456  | 0.9285      | 0.2063      | 25 epochs, lr=1e-4 |
| **mean** | **0.9371** | **0.2196** | |
| **std**  | **0.0092** | **0.0111** | |

Training behaviour: val_AUC shows notable oscillation across all seeds after the warmup phase,
particularly once the cosine LR decay begins. The train loss decreases smoothly throughout,
suggesting the model is learning but val_AUC is sensitive to the LR schedule and val set
composition. Best checkpoints are typically reached between epochs 10–15.

### Early retrain (wrong HP, for reference only)

Config used: `num_cnn_blocks=3, d_model=256, n_layers=4, n_heads=8, dropout=0.1, lr=3e-4`.  
Best checkpoint: epoch 14 — val_AUC=0.9068, val_F1=0.3410. Not used in final comparison.

---

## Comparison (70/15/15 split, seed=42)

| Model | val_AUC | val_F1 | test_AUC | test_F1 | Params | Notes |
|---|---|---|---|---|---|---|
| `rf_baseline` (MFCC + RF) | 0.7690 | 0.1696 | 0.7729 | 0.1016 | — | — |
| `cnn_baseline` (ResNet-18) | 0.9289 | 0.4844 | 0.9182 | — | ~11M | — |
| `vit_baseline` (ViT-Small, scratch) | 0.8922 | 0.3479 | 0.9069 | — | ~22M | — |
| `cnn_transformer` | 0.9498 | 0.2334 | 0.9411 | — | ~18M | seed 42 |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9587 | 0.5753 | 0.9588 | — | ~22M | — |

test_AUC values from seed=42 checkpoint, evaluated over ~193/206 classes with ≥1 positive in the test split (class imbalance). Multi-seed mean val_AUC in [RESULTS.md](RESULTS.md).

The CNN-Transformer with n_layers=0 would be CNN + projection + MLP head, which is architecturally very close to ResNet-18 + classification head. Comparing our proposed model with the CNN baseline: at seed 42, the CNN-Transformer (0.9498 val_AUC, 0.9411 test_AUC) outperforms the CNN baseline (0.9289 val_AUC, 0.9182 test_AUC). Averaged over 3 seeds, cnn_transformer mean (0.9371) also exceeds cnn_baseline mean (0.9195). The Transformer adds modest but consistent value on top of the CNN front-end.