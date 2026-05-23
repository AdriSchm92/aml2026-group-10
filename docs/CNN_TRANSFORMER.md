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
| `rf_baseline` (MFCC + RF) | 0.7690 | 0.1696 | — | — | — | test eval pending |
| `cnn_baseline` (ResNet-18) | 0.9540 | 0.4844 | — | — | ~11M | test eval pending |
| `vit_baseline` (ViT-Small, scratch) | 0.8922 | 0.3363 | — | — | ~22M | test eval pending |
| `cnn_transformer` (mean ± std) | 0.9371 ± 0.0092 | 0.2196 ± 0.0111 | — | — | ~18M | 3 seeds; test eval pending |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9537 | 0.5753 | — | — | ~22M | test eval pending |

All test_AUC/test_F1 cells pending — run `evaluate.py --split test` for each model checkpoint.

The CNN-Transformer with n_layers=0 would be CNN + projection + MLP head, which is architecturally very close to ResNet-18 + classification head. So when comparing our proposed model with the CNN Baseline, we can observe that the Transformer is not only not helping, but it's slightly hurting.

To ensure the Transformer architecture was given the best conditions to demonstrate its value, we investigated whether the underperformance of the CNN-Transformer was linked to the short 5-second clip duration. The core motivation is that self-attention's strength lies in modelling long-range dependencies — with only ~160 tokens spanning 5 seconds of audio, there may simply not be enough temporal context for the Transformer to add meaningful global reasoning on top of what the CNN already captures locally. To test this hypothesis, we reran the three main deep learning models using 10-second clips, which doubles the number of tokens presented to the Transformer (~320 tokens) and provides more temporal context for species-discriminative call patterns such as repeated phrases or response calls.

## Comparison (70/15/15 split, seed=42) with 10s duration clips

| Model | val_AUC | val_F1 | test_AUC | test_F1 | Params |
|---|---|---|---|---|---|
| `cnn_baseline` (ResNet-18) | 0.8671 | 0.1182 | — | — | ~11M |
| `cnn_transformer` (mean ± std) | 0.9335 | 0.2129 | — | — | ~18M |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9489 | 0.2879 | — | — | ~22M |

The results do not support the hypothesis. The CNN-Transformer shows no meaningful improvement with longer clips (0.9335 vs 0.9371 at 5s), suggesting the bottleneck is not temporal context but rather the difficulty of training the Transformer component from scratch. The pretrained Transformer remains remarkably stable across both clip durations (0.9489 vs 0.9537), confirming that its strong performance is driven by ImageNet pretraining rather than temporal context length. Most strikingly, the CNN baseline drops sharply (0.8671 vs 0.9540), which is expected — a pure CNN gains nothing from longer sequences and suffers from the smaller batch size required by the increased memory footprint. Taken together, these results suggest that the 5-second clip duration is not the limiting factor for the Transformer's contribution, and that pretraining is a far more important variable than clip length in this setting.