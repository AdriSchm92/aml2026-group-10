# RF Baseline Results (MFCC + Random Forest)

**Role in project:** "Simple baseline" — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#baselines)

Architecture: MFCC mean+std over time → `(2 × n_mfcc,)` feature vector per 5s chunk.  
`OneVsRestClassifier(RandomForestClassifier)` — one RF per class (206 binary classifiers).  
No deep learning, no spectrogram. Weakest expected baseline.

## HP Grid

12 combos: `n_mfcc ∈ {20, 40, 80}` × `n_estimators ∈ {100, 300}` × `max_depth ∈ {None, 20}`

Features cached per `(split, n_mfcc)` under `.cache/birdclef/` — re-extraction skipped on subsequent runs.

## Results

8/12 combos completed (n_mfcc=80 rows interrupted by session limit).

| n_mfcc | n_estimators | max_depth | val_AUC | val_F1 | sc_AUC | fit_time (s) |
|--------|--------------|-----------|---------|--------|--------|-------------|
| 20 | 100 | None | 0.7138 | 0.1418 | 0.4732 | 4398 |
| 20 | 100 | 20 | 0.7437 | 0.1397 | 0.5001 | 4304 |
| 20 | 300 | None | 0.7501 | 0.1464 | 0.4882 | 12949 |
| **20** | **300** | **20** | **0.7653** | **0.1433** | **0.4897** | **12589** |
| 40 | 100 | None | 0.7066 | 0.1338 | 0.4688 | 6189 |
| 40 | 100 | 20 | 0.7408 | 0.1319 | 0.4650 | 5994 |
| 40 | 300 | None | 0.7443 | 0.1363 | 0.4736 | 18377 |
| 40 | 300 | 20 | 0.7629 | 0.1353 | 0.4780 | 17839 |

**Best:** n_mfcc=20, n_estimators=300, max_depth=20 — **val_AUC 0.7653, val_F1 0.1433**

**Best checkpoint:** `best_rf_baseline.joblib`

## Launch command

```bash
# Full HP grid
python train_rf.py --tune --n_jobs 10 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs

# Single run (default HPs)
python train_rf.py --n_mfcc 40 --n_estimators 100
```

## Comparison vs other baselines

| Model | val_AUC | val_F1 | sc_AUC | Type |
|---|---|---|---|---|
| `rf_baseline` (MFCC + RF) | 0.7653 | 0.1433 | 0.4897 | No DL |
| `cnn_baseline` (ResNet-18) | 0.9570 | 0.5331 | — | Pure CNN |
| `vit_baseline` (ViT-Small) | 0.9059 | 0.4060 | — | Pure Transformer |
| `cnn_transformer` | _pending_ | _pending_ | _pending_ | Hybrid |

Expected: RF substantially below CNN/ViT — establishes the value of deep feature learning over handcrafted MFCC.
