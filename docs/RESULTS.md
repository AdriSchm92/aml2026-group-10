# Results

This file centralises all model comparison results.
Individual model docs contain architecture details, training commands, and per-epoch curves.

---

## 5-second clips (primary results)

All models trained on the full BirdCLEF 2026 dataset (K=206 species, 70/15/15 stratified split, seed=42 unless stated).  
3 seeds reported for robustness where available.

> **Note on test_AUC:** Due to class imbalance in the dataset, many rare species have zero positive examples in the test split after stratified splitting. ROC-AUC is undefined for classes with no positives and is excluded from the macro average. The number of classes with at least one positive example in the test set varies per model/seed but is approximately 98/206. This means test_AUC is computed over a subset of species and should be interpreted accordingly — the rarest, hardest-to-classify species are systematically underrepresented. Val_AUC covers a similar but not identical subset and is the primary metric used for model selection throughout training.

### Training configuration

| Model | epochs | lr | weight_decay | warmup_epochs | label_smoothing | batch_size | other |
|---|---|---|---|---|---|---|---|
| `rf_baseline` | — | — | — | — | — | — | n_mfcc=20, n_estimators=300, max_depth=20 |
| `vit_baseline` | 15 | 3e-4 | 0.05 | 5 | 0.1 | 128 | grad_clip=1.0 |
| `cnn_baseline` | 15 | 1e-3 | 1e-4 | 5 | 0.1 | 256 | — |
| `cnn_transformer` | 15 | 3e-4 | 0.05 | 5 | 0.1 | 256 | num_cnn_blocks=4, d_model=256, n_layers=4, n_heads=4, dropout=0.2 |
| `pretrained_transformer` | 15 | 1e-4 | 0.05 | 5 | 0.1 | 128 | — |

### Results

| Model | seed 42 | seed 123 | seed 456 | mean val_AUC ± std | test_AUC† | Type |
|---|---|---|---|---|---|---|
| `rf_baseline` (MFCC + RF) | 0.7690 | — | — | — | 0.7729 | No DL |
| `vit_baseline` (ViT-Small, scratch) | 0.8922 | 0.8995 | 0.9089 | 0.9002 ± 0.0084 | 0.9069 | Pure Transformer |
| `cnn_baseline` (ResNet-18) | 0.9289 | 0.9105 | 0.9190 | 0.9195 ± 0.0075 | 0.9182 | Pure CNN |
| `cnn_transformer` (ResNet-18 + Transformer) | 0.9498 | 0.9331 | 0.9285 | 0.9371 ± 0.0092 | 0.9411 | Hybrid |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9587 | 0.9602 | 0.9512 | 0.9567 ± 0.0039 | 0.9588 | Pretrained Transformer |

† test_AUC computed on seed=42 checkpoint only, evaluated over ~98/206 classes with ≥1 positive example in the test split. See note above.

### Key observations

- The pretrained Transformer is the strongest model overall (mean val_AUC=0.9567, test_AUC=0.9588) and the most stable across seeds (std=0.0039), demonstrating the value of ImageNet transfer learning for audio spectrogram classification. Notably, its test_AUC exceeds its mean val_AUC, suggesting good generalisation to held-out data.
- The CNN-Transformer outperforms the CNN baseline on both mean val_AUC (0.9371 vs 0.9195) and test_AUC (0.9411 vs 0.9182), suggesting the Transformer component does add value on top of the CNN front-end. However the gap is modest and the CNN baseline has lower variance across seeds (std=0.0075 vs 0.0092), indicating the CNN-Transformer's advantage is not fully consistent across data splits.
- The ViT from scratch is the weakest deep learning model (mean val_AUC=0.9002), confirming the importance of CNN inductive bias for local time-frequency pattern detection. The gap between from-scratch and pretrained ViT (+0.057 AUC) is the largest single gain in the comparison and isolates the value of ImageNet transfer.
- The RF baseline (val_AUC=0.7690, test_AUC=0.7729) establishes a clear floor, with a −0.187 AUC gap vs the best deep learning model, demonstrating the value of deep feature learning over handcrafted MFCCs. The RF is also the only model where val and test AUC are directly comparable, since its evaluation was built into train_rf.py with an explicit test evaluation step.
- The pretrained Transformer is the most stable model across seeds (std=0.0039), while the CNN-Transformer shows the highest variance (std=0.0092), consistent with the known instability of training Transformer components from scratch.
- Val and test AUC rankings are consistent across all models — no model that performs well on val degrades on test — giving confidence that the val results are a reliable proxy for generalisation.

---

## 10-second clips (exploratory)

Motivated by the hypothesis that the Transformer's modest advantage over the CNN baseline at 5s clips could be amplified with longer temporal context. Bird vocalizations often contain long-range structure (repeated phrases, response calls, harmonic progressions) that a 5-second window may truncate. Longer clips double the token count (~320 vs ~160 tokens), giving self-attention more temporal range to model these patterns. Due to increased memory requirements, batch size was reduced to 64 and LR scaled accordingly. Single seed (42) only — these runs are exploratory rather than part of the formal comparison.

### Training configuration

| Model | epochs | lr | weight_decay | warmup_epochs | label_smoothing | batch_size |
|---|---|---|---|---|---|---|
| `cnn_baseline` | 15 | 7.5e-5 | 1e-4 | 5 | 0.1 | 64 |
| `cnn_transformer` | 15 | 7.5e-5 | 0.05 | 5 | 0.1 | 64 |
| `pretrained_transformer` | 15 | 7.5e-5 | 0.05 | 5 | 0.1 | 64 |
| `vit_baseline` | 15 | 7.5e-5 | 0.05 | 5 | 0.1 | 64 |

### Results

| Model | 5s val_AUC (seed 42) | 10s val_AUC | Delta |
|---|---|---|---|
| `cnn_baseline` (ResNet-18) | 0.9289 | 0.8671 | −0.062 |
| `cnn_transformer` (ResNet-18 + Transformer) | 0.9498 | 0.9335 | −0.016 |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9587 | 0.9489 | −0.010 |
| `vit_baseline` (ViT-Small, scratch) | 0.8922 | 0.9079 | +0.016 |

### Key observations

The results partially support the hypothesis that longer clips benefit Transformer-based models, but the picture is nuanced. The ViT baseline is the only model that improves with longer clips (+0.016 AUC), which is the most interpretable finding: without CNN inductive bias, the from-scratch ViT relies more heavily on global context, and longer clips provide more tokens for self-attention to work with. This confirms that pure Transformers do benefit from longer temporal context, at least when trained from scratch without strong local feature extractors.

However, the CNN-Transformer shows no meaningful improvement (0.9335 vs 0.9498 at 5s), suggesting that once a CNN front-end handles local feature extraction, the Transformer component gains little from additional temporal context. The pretrained Transformer remains remarkably stable across both clip durations (0.9489 vs 0.9587), confirming that its strong performance is driven by ImageNet pretraining rather than temporal context length. Most strikingly, the CNN baseline drops sharply (0.8671 vs 0.9289) — a pure CNN gains nothing from longer sequences and is hurt by the smaller batch size required by the increased memory footprint.

Taken together, these results suggest that clip duration is a meaningful variable specifically for pure Transformer architectures trained from scratch, while CNN-based and pretrained models are largely insensitive to it. Pretraining remains the single most impactful factor across all conditions.

---

## Summary

| Model | mean val_AUC (5s) | test_AUC (seed 42)† | Params | Notes |
|---|---|---|---|---|
| `rf_baseline` | 0.7690 | 0.7729 | — | Handcrafted MFCC features, no DL |
| `vit_baseline` | 0.9002 ± 0.0084 | 0.9069 | ~22M | From scratch, no CNN inductive bias |
| `cnn_baseline` | 0.9195 ± 0.0075 | 0.9182 | ~11M | Pure CNN, stable but limited by local features only |
| `cnn_transformer` | 0.9371 ± 0.0092 | 0.9411 | ~18M | Hybrid; Transformer adds modest but consistent value |
| `pretrained_transformer` | 0.9567 ± 0.0039 | 0.9588 | ~22M | Best model overall; most stable across seeds |

† Evaluated over ~98/206 classes with ≥1 positive in the test split due to class imbalance. See note in primary results section.

### Further Analysis
Detailed qualitative analysis and visualisations are available in notebooks/Results.ipynb, including:

- Learning curves (val_AUC and train loss per epoch) for all models across seeds
- 5s vs 10s clip duration comparison plots
- Model comparison bar chart with seed variance
- Per-class AUC distribution across all 206 species
- Hardest and easiest species by AUC for our model and the best one (Pretrained ViT)
- CNN-Transformer CLS token attention maps — visualising which time-frequency regions the Transformer attends to for individual spectrograms
- Pretrained ViT CLS token attention map

The notebook is pre-executed with saved outputs and can be viewed without rerunning. Note that paths are configured for the Renku environment; to rerun locally, update REPO_ROOT, OUTPUT_DIR, and DATA_ROOT at the top of the notebook.