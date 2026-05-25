# Results

This file centralises all model comparison results.
Individual model docs contain architecture details, training commands, and per-epoch curves.

---

## 5-second clips (primary results)

All models trained on the full BirdCLEF 2026 dataset (K=206 species, 70/15/15 stratified split, seed=42 unless stated).  
3 seeds reported for robustness where available. Test evaluations pending for all deep learning models.

### Training configuration

| Model | epochs | lr | weight_decay | warmup_epochs | label_smoothing | batch_size | other |
|---|---|---|---|---|---|---|---|
| `rf_baseline` | — | — | — | — | — | — | n_mfcc=20, n_estimators=300, max_depth=20 |
| `vit_baseline` | 15 | 3e-4 | 0.05 | 5 | 0.1 | 128 | grad_clip=1.0 |
| `cnn_baseline` | 15 | 1e-3 | 1e-4 | 5 | 0.1 | 256 | — |
| `cnn_transformer` | 15 | 3e-4 | 0.05 | 5 | 0.1 | 256 | num_cnn_blocks=4, d_model=256, n_layers=4, n_heads=4, dropout=0.2 |
| `pretrained_transformer` | 15 | 1e-4 | 0.05 | 5 | 0.1 | 128 | — |

### Results
| Model | seed 42 | seed 123 | seed 456 | mean ± std | test_AUC | Type |
|---|---|---|---|---|---|---|
| rf_baseline (MFCC + RF) | 0.7690 | — | — | — | 0.7729 | No DL |
| vit_baseline (ViT-Small, scratch) | 0.8922 | 0.8995 | 0.9089 | 0.9002 ± 0.0084 | — | Pure Transformer |
| cnn_baseline (ResNet-18) | 0.9289 | 0.9105 | 0.9190 | 0.9195 ± 0.0075 | — | Pure CNN |
| cnn_transformer (ResNet-18 + Transformer) | 0.9498 | 0.9331 | 0.9285 | 0.9371 ± 0.0092 | — | Hybrid |
| pretrained_transformer (ViT-Small, ImageNet) | 0.9587 | 0.9602 | 0.9512 | 0.9567 ± 0.0039 | — | Pretrained Transformer |

### Key observations

- The pretrained Transformer is the strongest model overall (mean val_AUC=0.9567) and the most stable across seeds (std=0.0039), demonstrating the value of ImageNet transfer learning for audio spectrogram classification.
- The CNN-Transformer outperforms the CNN baseline on mean val_AUC (0.9371 vs 0.9195), suggesting the Transformer component does add some value on top of the CNN front-end. However the gap is modest and the CNN baseline has lower variance (std=0.0075 vs 0.0092), indicating the CNN-Transformer's advantage is not consistent across data splits.
- The ViT from scratch is the weakest deep learning model (mean val_AUC=0.9002), confirming the importance of CNN inductive bias for local time-frequency pattern detection. The gap between from-scratch and pretrained ViT (+0.057 AUC) is the largest single gain in the comparison and isolates the value of ImageNet transfer.
- The RF baseline (0.7690) establishes a clear floor, with a −0.187 AUC gap vs the best deep learning model, demonstrating the value of deep feature learning over handcrafted MFCCs.
- The pretrained Transformer is the most stable model across seeds (std=0.0039), while the CNN-Transformer shows the highest variance (std=0.0092), consistent with the known instability of training Transformer components from scratch.

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

| Model | 5s val_AUC (seed 42)| 10s val_AUC | Delta |
|---|---|---|---|
| `cnn_baseline` (ResNet-18) | 0.9540 | 0.8671 | −0.087 |
| `cnn_transformer` (ResNet-18 + Transformer) | 0.9498 | 0.9335 | −0.016 |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9537 | 0.9489 | −0.005 |
| `vit_baseline` (Standard Vision Transformer) | 0.8922 | 0.9079 | +0.157 |


### Key observations

The results partially support the hypothesis that longer clips benefit Transformer-based models, but the picture is nuanced. The ViT baseline is the only model that improves with longer clips (+0.016 AUC), which is the most interpretable finding: without CNN inductive bias, the from-scratch ViT relies more heavily on global context, and longer clips provide more tokens for self-attention to work with. This confirms that pure Transformers do benefit from longer temporal context, at least when trained from scratch without strong local feature extractors.

However, the CNN-Transformer shows no meaningful improvement (0.9335 vs 0.9498 at 5s), suggesting that once a CNN front-end handles local feature extraction, the Transformer component gains little from additional temporal context. The pretrained Transformer remains remarkably stable across both clip durations (0.9587 vs 0.9537), confirming that its strong performance is driven by ImageNet pretraining rather than temporal context length. Most strikingly, the CNN baseline drops sharply (0.8671 vs 0.9289) — a pure CNN gains nothing from longer sequences and is hurt by the smaller batch size required by the increased memory footprint.

Taken together, these results suggest that clip duration is a meaningful variable specifically for pure Transformer architectures, while CNN-based models are largely insensitive to it. Pretraining remains the single most impactful factor across all conditions.

---

## Summary

| Model | mean val_AUC (5s) | Params | Notes |
|---|---|---|---|
| `rf_baseline` | 0.7690 | — | Handcrafted MFCC features, no DL |
| `vit_baseline` | 0.9002 ± 0.0084 | ~22M | From scratch, no CNN inductive bias |
| `cnn_baseline` | 0.9195 ± 0.0075 | ~11M | Pure CNN, high variance across seeds |
| `cnn_transformer` | 0.9371 ± 0.0092 | ~18M | Transformer not contributing meaningfully |
| `pretrained_transformer` | 0.9567 ± 0.0039 | ~22M | Best model, most stable |

All test_AUC cells pending — run `evaluate.py --split test` for each model checkpoint.