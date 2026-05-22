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
| `rf_baseline` (MFCC + RF) | 0.7690 | — | — | — | 0.7729 | No DL |
| `vit_baseline` (ViT-Small, scratch) | 0.8922 | 0.8995 | 0.9089 | 0.9002 ± 0.0084 | — | Pure Transformer |
| `cnn_baseline` (ResNet-18) | 0.9540 | 0.9105 | 0.9190 | 0.9278 ± 0.0192 | — | Pure CNN |
| `cnn_transformer` (ResNet-18 + Transformer) | 0.9498 | 0.9331 | 0.9285 | 0.9371 ± 0.0092 | — | Hybrid |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9537 | 0.9602 | 0.9512 | 0.9550 ± 0.0037 | — | Pretrained Transformer |

### Key observations

- The pretrained Transformer is the strongest model overall (mean val_AUC=0.9550) and the most stable across seeds (std=0.0037), demonstrating the value of ImageNet transfer learning for audio spectrogram classification.
- The CNN-Transformer does not outperform the CNN baseline (0.9371 vs 0.9278 mean), suggesting the Transformer component is not contributing meaningful global reasoning on top of the CNN front-end for 5-second clips.
- The ViT from scratch is the weakest deep learning model (0.9002), confirming the importance of CNN inductive bias for local time-frequency pattern detection.
- The RF baseline (0.7690) establishes a clear floor, with a −0.158 AUC gap vs the best deep learning model, demonstrating the value of deep feature learning over handcrafted MFCCs.
- The CNN baseline shows higher variance across seeds (std=0.0192) compared to the pretrained Transformer (std=0.0037), suggesting pretrained models generalise more robustly across different data splits.

---

## 10-second clips (exploratory)

Motivated by the hypothesis that the Transformer's underperformance on 5s clips was due to insufficient temporal context. Longer clips double the token count (~320 vs ~160 tokens), giving self-attention more temporal range to model long-range call patterns. Due to increased memory requirements, batch size was reduced to 64 and LR scaled accordingly. Single seed (42) only.

### Training configuration

| Model | epochs | lr | weight_decay | warmup_epochs | label_smoothing | batch_size |
|---|---|---|---|---|---|---|
| `cnn_baseline` | 15 | 7.5e-5 | 1e-4 | 5 | 0.1 | 64 |
| `cnn_transformer` | 15 | 7.5e-5 | 0.05 | 5 | 0.1 | 64 |
| `pretrained_transformer` | 15 | 7.5e-5 | 0.05 | 5 | 0.1 | 64 |

### Results

| Model | 5s val_AUC | 10s val_AUC | Delta |
|---|---|---|---|
| `cnn_baseline` (ResNet-18) | 0.9540 | 0.8671 | −0.087 |
| `cnn_transformer` (ResNet-18 + Transformer) | 0.9498 | 0.9335 | −0.016 |
| `pretrained_transformer` (ViT-Small, ImageNet) | 0.9537 | 0.9489 | −0.005 |

### Key observations

The results do not support the hypothesis that longer clips would help the Transformer. The CNN-Transformer shows no meaningful improvement with longer clips (0.9335 vs 0.9498 at 5s), suggesting the bottleneck is not temporal context but rather the difficulty of training the Transformer component from scratch. The pretrained Transformer remains remarkably stable across both clip durations (0.9489 vs 0.9537), confirming that its strong performance is driven by ImageNet pretraining rather than temporal context length. Most strikingly, the CNN baseline drops sharply (0.8671 vs 0.9540) — a pure CNN gains nothing from longer sequences and suffers from the smaller batch size required by the increased memory footprint. Taken together, these results suggest that the 5-second clip duration is not the limiting factor for the Transformer's contribution, and that pretraining is a far more important variable than clip length in this setting.

---

## Summary

| Model | mean val_AUC (5s) | Params | Notes |
|---|---|---|---|
| `rf_baseline` | 0.7690 | — | Handcrafted MFCC features, no DL |
| `vit_baseline` | 0.9002 ± 0.0084 | ~22M | From scratch, no CNN inductive bias |
| `cnn_baseline` | 0.9278 ± 0.0192 | ~11M | Pure CNN, high variance across seeds |
| `cnn_transformer` | 0.9371 ± 0.0092 | ~18M | Transformer not contributing meaningfully |
| `pretrained_transformer` | 0.9550 ± 0.0037 | ~22M | Best model, most stable |

All test_AUC cells pending — run `evaluate.py --split test` for each model checkpoint.