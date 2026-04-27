# RF Baseline Results (MFCC + Random Forest)

**Role in project:** "Simple baseline" — see [PROBLEMSETTING.md](PROBLEMSETTING.md#baselines)

Architecture: MFCC mean+std over time → `(2 × n_mfcc,)` feature vector per 5s chunk.  
`OneVsRestClassifier(RandomForestClassifier)` — one RF per class (206 binary classifiers).  
No deep learning, no spectrogram. Weakest expected baseline.

## HP Grid

12 combos: `n_mfcc ∈ {20, 40, 80}` × `n_estimators ∈ {100, 300}` × `max_depth ∈ {None, 20}`

Features cached per `(split, n_mfcc)` under `.cache/birdclef/` — re-extraction skipped on subsequent runs.

## Results

_Results pending — run in progress._

| n_mfcc | n_estimators | max_depth | val_AUC | val_F1 | sc_AUC | fit_time (s) |
|--------|-------------|-----------|---------|--------|--------|-------------|
| | | | | | | |

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
| `rf_baseline` (MFCC + RF) | _pending_ | _pending_ | _pending_ | No DL |
| `cnn_baseline` (ResNet-18) | 0.9570 | 0.5331 | — | Pure CNN |
| `vit_baseline` (ViT-Small) | 0.9059 | 0.4060 | — | Pure Transformer |
| `cnn_transformer` | _pending_ | _pending_ | _pending_ | Hybrid |

Expected: RF substantially below CNN/ViT — establishes the value of deep feature learning over handcrafted MFCC.
