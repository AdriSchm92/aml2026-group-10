# Training

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
# GPU (CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Data

```bash
python scripts/stash_birdclef_data.py --kaggle-download
```

Data and trained model weights are stored persistently on **SwitchDrive**:
`https://drive.switch.ch/index.php/s/9cIBdwamXXtGHCQ`  
Upload checkpoints (`best_*.pt`) and results (`metrics_*.jsonl`, `test_results_*.json`)
there after each training run so the team can share them without re-running.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `KAGGLE_API_TOKEN` | Kaggle API token for `--kaggle-download` | `~/.kaggle/access_token` |
| `DATA_ROOT` | Path to data root (`train.csv`, `train_audio/`, etc.) | auto-detected |
| `BIRDCLEF_STASH_DIR` | Override local stash path | `./birdclef_stash` |
| `BIRDCLEF_DURATION_CACHE` | Path to duration cache file | `.cache/birdclef/` |
| `BIRDCLEF_SPEC_CACHE` | Path to precomputed spectrogram cache dir (see `scripts/precompute_specs.py`) | — |
| `TRAINING_OUTPUT_DIR` | Where checkpoints and metrics are written | auto-detected |
| `TRAINING_OUTPUT_SUBDIR` | Subdir name under `kaggle-data/` on Renku | `aml2026-group10-runs` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot for training notifications | — |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for notifications | — |

Data root resolution order: `--data_root` CLI → `DATA_ROOT` env → local stash → `/kaggle/input/birdclef-2026` → `data/raw/`.

---

## Running training

```bash
# Full run (default: 5 epochs, batch 64)
python train.py --model cnn_baseline

# Renku GPU — save to SwitchDrive
python train.py --model cnn_baseline \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs

# Smoke test
python train.py --model cnn_baseline --epochs 1 --batch_size 8 --limit_train_batches 4 --limit_val_batches 4
```

Key CLI args: `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--val_size`, `--test_size`, `--seed`, `--grad_clip`, `--tag` (checkpoint name suffix), `--no-amp` (disable mixed precision).

### Data split args

| Arg | Default | Description |
|---|---|---|
| `--val_size` | 0.15 | Fraction of recordings held out for validation |
| `--test_size` | 0.15 | Fraction of recordings held out for final test (built but never touched during training) |
| `--data_subset_min_recordings N` | None | Restrict to species with ≥ N recordings (K=69 at N=200). Use for HP search or quick experiments. |

All models should use the same `--val_size`, `--test_size`, and `--seed` for results to be directly comparable.

### Architecture kwargs (`--model_kwargs`)

For `cnn_transformer`, architecture hyperparameters can be passed as a JSON string:

```bash
python train.py --model cnn_transformer \
    --model_kwargs '{"d_model": 256, "n_layers": 4, "n_heads": 8, "num_cnn_blocks": 3, "dropout": 0.1}'
```

`--model_kwargs` is forwarded to `build_model(**kwargs)` via `models/registry.py`. Existing models accept and ignore unknown kwargs.

### Per-model recommended hyperparameters

| Model | `--lr` | `--weight_decay` | `--warmup_epochs` | `--label_smoothing` | Notes |
|---|---|---|---|---|---|
| `rf_baseline` | — | — | — | — | sklearn, see `train_rf.py` |
| `cnn_baseline` | `1e-3` | `1e-4` | `0` | `0.0` | CosineAnnealingLR, no warmup needed |
| `vit_baseline` | `3e-4` | `0.05` | `5` | `0.1` | Transformer from scratch; warmup critical |
| `cnn_transformer` (from scratch) | `3e-4` | `0.05` | `5` | `0.1` | Explicitly pass `--lr 3e-4`; train.py default is `1e-3` |
| `cnn_transformer` (pretrained CNN) | `1e-4` | `0.05` | `5` | `0.1` | Lower LR; pretrained weights need gentle fine-tuning |
| `pretrained_transformer` | `1e-4` | `0.05` | `5` | `0.1` | Fine-tuning; warmup protects ImageNet weights |

### CNN warm-start (`cnn_transformer` only)

When `--model cnn_transformer`, the CNN front-end is automatically warm-started from `best_cnn_baseline.pt` in the output dir if the file exists. This gives faster convergence with a principled initialisation.

| Arg / setting | Behaviour |
|---|---|
| _(default)_ | Auto-detect `best_cnn_baseline.pt` in output dir |
| `--init_from <path>` | Explicit checkpoint path |
| `--no_init` | Disable warm-start entirely |
| `--model_kwargs '{"pretrained_cnn": true}'` | Auto-skips warm-start; CNN already has ImageNet weights |

---

## Evaluating a checkpoint

```bash
# Val set (threshold tuning on val — exploratory)
python evaluate.py --model cnn_transformer

# Test set (val-tuned thresholds applied to held-out test — final reporting)
python evaluate.py --model cnn_transformer --split test
```

`--split test` reads the per-class thresholds saved in the checkpoint and applies them to the test set without retuning. The result is written to `test_results_<model>.json`.

---

## HP search

HP search runs each trial on the reduced K=69 subset (`--data_subset_min_recordings 200`) for ~3 epochs to evaluate structural choices cheaply, then a final full-data retrain uses the best config.

```bash
# Default: 6 trials, ≤8h budget, cnn_transformer config
python scripts/hp_search.py --model cnn_transformer

# Custom
python scripts/hp_search.py --model cnn_transformer \
    --n_trials 8 --max_hours 10 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs
```

HP grids are defined in `configs/hp_<model>.yaml`. Results are logged to `hp_results_<model>.jsonl`.

After HP search, the final retrain uses the best config on full data:

```bash
python train.py --model cnn_transformer \
    --model_kwargs '{"num_cnn_blocks": 4, "d_model": 256, "n_layers": 4, "n_heads": 4, "dropout": 0.2}' \
    --epochs 15 --lr 3e-4 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1 \
    --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs
```

---

## Adding a new model

1. Create `models/<name>.py` with one function:

```python
import torch.nn as nn

def build_model(num_classes: int, **_ignored) -> nn.Module:
    return MyModel(num_classes=num_classes)
```

2. Run immediately — no other changes needed:

```bash
python train.py --model <name>
```

Registry auto-discovers all files in `models/` (skips `registry.py`, `__init__.py`).

### Current models

| Name | Description | Status |
|---|---|---|
| `cnn_baseline` | ResNet-18 on mel-spectrograms | done — 3 seeds, HP search complete — [CNN_BASELINE.md](CNN_BASELINE.md) |
| `vit_baseline` | Pure ViT (from scratch) on raw spectrogram patches | done — 3 seeds, HP search complete — [VIT_BASELINE.md](VIT_BASELINE.md) |
| `rf_baseline` | MFCC + Random Forest | done — 12-combo HP grid, test eval complete — [RF_BASELINE.md](RF_BASELINE.md) |
| `cnn_transformer` | CNN front-end + Transformer encoder (main model) | done — 3 seeds, HP search complete — [CNN_TRANSFORMER.md](CNN_TRANSFORMER.md) |
| `pretrained_transformer` | Pretrained ViT fine-tuned on spectrograms | done — 3 seeds — [PRETRAINED_TRANSFORMER.md](PRETRAINED_TRANSFORMER.md) |
