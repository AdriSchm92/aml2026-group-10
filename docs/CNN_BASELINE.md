# CNN Baseline Results (ResNet-18)

**Role in project:** "ML baseline" — see [PROBLEMSETTING.md](../PROBLEMSETTING.md#baselines)

Architecture: ResNet-18 (`timm`, `in_chans=1`) on log-mel spectrograms `(1, 128, 313)`.  
Loss: BCEWithLogitsLoss. Optimizer: AdamW (lr=1e-3, wd=1e-4). Scheduler: CosineAnnealingLR.  
Data: full BirdCLEF 2026 train set, **70/15/15 stratified split** (seed=42), SpecAugment enabled.  
Spectrogram cache enabled (`--spec_cache_dir /tmp/birdclef_specs`) — ~255s/epoch.

## Training Run

| Epoch | train_loss  | val_AUC    | val_F1     | time (s) | LR       |
| ----- | ----------- | ---------- | ---------- | -------- | -------- |
| 1     | 0.02875     | 0.8598     | 0.1751     | 267      | 9.045e-4 |
| 2     | 0.01714     | 0.9302     | 0.3564     | 254      | 6.545e-4 |
| 3     | 0.01284     | 0.9463     | 0.4252     | 254      | 3.455e-4 |
| 4     | 0.01003     | 0.9525     | 0.4653     | 254      | 9.549e-5 |
| **5** | **0.00837** | **0.9540** | **0.4844** | 255      | 0.000e+0 |

Best checkpoint: epoch 5 — **val_AUC 0.9540, val_F1 0.4844**

AUC still slowly improving at epoch 5 (cosine LR just reached zero). Additional epochs with a longer schedule would likely push further. The ~255s/epoch reflects cached spectrograms; raw-decode runs take ~9200s/epoch.

## HP Search

HP grid: `configs/hp_cnn_baseline.yaml` — covers `resnet_variant` (ResNet-18 vs ResNet-34), `lr`, `weight_decay`.  
6 trials × 3 epochs on K=69 subset (`min_recordings ≥ 200`).

```bash
python scripts/hp_search.py --model cnn_baseline --n_trials 6 \
    --data_root $DATA_ROOT --spec_cache_dir $BIRDCLEF_SPEC_CACHE
```

### HP search results (K=69 subset, 3 epochs/trial)

| Trial | `resnet_variant` | `lr` | `weight_decay` | val_AUC |
|---|---|---|---|---|
| 0 | resnet34 | 1e-3 | 1e-5 | 0.9578 |
| **1** | **resnet34** | **3e-3** | **1e-4** | **0.9640** |
| 2 | resnet34 | 1e-3 | 1e-4 | 0.9593 |
| 3 | resnet34 | 3e-3 | 1e-5 | 0.9579 |
| 4 | resnet18 | 1e-3 | 1e-5 | 0.9539 |
| 5 | resnet18 | 3e-3 | 1e-4 | 0.9531 |

**Best config:** Trial 1 — `resnet34, lr=3e-3, weight_decay=1e-4` → val_AUC 0.9640

Key observations: ResNet-34 consistently outperforms ResNet-18 across all LR/wd combinations (~+0.005 AUC). Higher LR (3e-3) works well for CNNs unlike Transformers where it causes collapse. Deeper backbone captures richer mel-spectrogram features at similar training cost (~975s vs ~615s per trial).

## Next comparison models

1. `vit_baseline` — pure ViT on raw patches, no CNN front-end — [docs/VIT_BASELINE.md](VIT_BASELINE.md)
2. `cnn_transformer` — CNN front-end + Transformer encoder (main proposed model) — [docs/CNN_TRANSFORMER.md](CNN_TRANSFORMER.md)