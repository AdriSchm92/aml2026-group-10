# RF Baseline Results (MFCC + Random Forest)
**Role in project:** "Simple baseline" — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#baselines)

Architecture: MFCC mean+std over time → `(2 × n_mfcc,)` feature vector per 5s chunk.  
`OneVsRestClassifier(RandomForestClassifier)` — one RF per class (206 binary classifiers).  
No deep learning, no spectrogram. Weakest expected baseline.

## HP Grid
12 combos: `n_mfcc ∈ {20, 40, 80}` × `n_estimators ∈ {100, 300}` × `max_depth ∈ {None, 20}`
Features cached per `(split, n_mfcc)` under `.cache/birdclef/` — re-extraction skipped on subsequent runs.
## Results

| n_mfcc | n_estimators | max_depth | val_AUC | val_F1 | sc_AUC | fit_time (s) |
|--------|-------------|-----------|---------|--------|--------|-------------|
| 20 | 100 | None | 0.7186 | 0.1676 | 0.4431 | 2440 |
| 20 | 100 | 20 | 0.7491 | 0.1650 | 0.5054 | 2393 |
| 20 | 300 | None | 0.7539 | 0.1740 | 0.4528 | 7101 |
| **20** | **300** | **20** | **0.7690** | **0.1696** | **0.4897** | **6902** |
| 40 | 100 | None | 0.7122 | 0.1594 | 0.4609 | 3473 |
| 40 | 100 | 20 | 0.7447 | 0.1555 | 0.5206 | 3340 |
| 40 | 300 | None | 0.7494 | 0.1663 | 0.4691 | 10014 |
| 40 | 300 | 20 | 0.7671 | 0.1646 | 0.4976 | 9632 |
| 80 | 100 | None | 0.6984 | 0.1459 | 0.4771 | 5701 |
| 80 | 100 | 20 | 0.7391 | 0.1413 | 0.4720 | 5352 |
| 80 | 300 | None | 0.7325 | 0.1490 | 0.4658 | 16476 |
| 80 | 300 | 20 | 0.7524 | 0.1464 | 0.4694 | 15509 |

**Best config:** `n_mfcc=20, n_estimators=300, max_depth=20` — val_AUC=0.7690, val_F1=0.1696, sc_AUC=0.4897  
**Best checkpoint:** `best_rf_baseline.joblib`

### Key observations
- More MFCCs (n_mfcc=80) consistently hurts performance vs n_mfcc=20 — higher coefficients add noise rather than useful information for bird species discrimination.
- `max_depth=20` outperforms unbounded trees across all combos — unconstrained depth overfits the MFCC features.
- F1 scores are very low across the board (0.14–0.17), reflecting the difficulty of 206-class imbalanced classification with handcrafted features.
- sc_AUC (~0.44–0.52) is substantially below val_AUC (~0.70–0.77), indicating poor domain generalisation from clean clips to real-world soundscapes.
- Fit time scales steeply with n_estimators and n_mfcc — 300 trees at n_mfcc=80 took ~4.5 hours.

## Launch command
```bash
# Full HP grid (12 combos)
python train_rf.py --tune --n_jobs 10 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs

# Best config (from HP search)
python train_rf.py --n_mfcc 20 --n_estimators 300 --max_depth 20

# Single run (code defaults — not the best config)
python train_rf.py --n_mfcc 40 --n_estimators 100
```

## Comparison vs other baselines

All seed-42 val values. Multi-seed means in [RESULTS.md](RESULTS.md).

| Model | val_AUC | val_F1 | sc_AUC | test_AUC | test_F1 | Type |
|---|---|---|---|---|---|---|
| `rf_baseline` (MFCC + RF) | 0.7690 | 0.1696 | 0.4897 | 0.7729 | 0.1016 | No DL |
| `cnn_baseline` (ResNet-18) | 0.9540 | 0.4844 | — | — | — | Pure CNN |
| `vit_baseline` (ViT-Small) | 0.8922 | 0.3479 | — | — | — | Pure Transformer |
| `cnn_transformer` | 0.9498 | 0.2334 | — | — | — | Hybrid |
| `pretrained_transformer` | 0.9537 | 0.5753 | — | — | — | Pretrained Transformer |

Expected: RF substantially below all deep learning models — establishes the value of deep feature learning over handcrafted MFCC. Delta vs best model (pretrained_transformer mean 0.9550): −0.186 AUC.