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

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `KAGGLE_API_TOKEN` | Kaggle API token for `--kaggle-download` | `~/.kaggle/access_token` |
| `DATA_ROOT` | Path to data root (`train.csv`, `train_audio/`, etc.) | auto-detected |
| `BIRDCLEF_STASH_DIR` | Override local stash path | `./birdclef_stash` |
| `BIRDCLEF_DURATION_CACHE` | Path to duration cache file | `.cache/birdclef/` |
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

Key CLI args: `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--val_size`, `--seed`, `--grad_clip`, `--tag` (checkpoint name suffix), `--no-amp` (disable mixed precision).

---

## Adding a new model

1. Create `models/<name>.py` with one function:

```python
import torch.nn as nn

def build_model(num_classes: int) -> nn.Module:
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
| `cnn_baseline` | ResNet-18 on mel-spectrograms | done — [CNN_BASELINE.md](CNN_BASELINE.md) |
| `vit_baseline` | Pure ViT on raw spectrogram patches | todo |
| `cnn_transformer` | CNN front-end + Transformer encoder | todo |
| `rf_baseline` | MFCC + Random Forest | todo |
