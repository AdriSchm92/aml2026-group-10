# Data Pipeline for BirdCLEF 2026

Source: `data/preprocessing/data_pipeline.py`

## Overview

Full data pipeline for BirdCLEF 2026. Handles loading, preprocessing, augmentation, and
batching from two data sources into PyTorch DataLoaders.

---

## Data Sources

### 1. `train_audio/` — Clean single-species recordings

- Sourced from Xeno-canto and iNaturalist, uploaded by birdwatchers
- Each file contains one primary species, recorded in relatively clean conditions
- Used for all three splits (train / val / test)
- Labels: single species per recording (`primary_label` column in `train.csv`)

### 2. `train_soundscapes/` — Real-world passive acoustic recordings

- Recorded from the same field locations and equipment as the hidden test set
- Contain multiple species vocalizing simultaneously over natural background noise
- Used for training only (not val/test) to close the domain gap
- Labels: expert-annotated, semicolon-separated species lists per 5-second segment
  (`train_soundscapes_labels.csv`)

---

## Preprocessing Pipeline

Each audio clip goes through the following steps in `__getitem__`:

| Step | Operation | Details |
| ---- | --------- | ------- |
| 1 | Load audio | 5-second window at `offset` seconds, resampled to 32000 Hz mono |
| 2 | Zero-pad | Short clips (< 5s) padded with silence on the right |
| 3 | Mel-spectrogram | Shape `(128, 313)`, hop=512, n_fft=1024, fmin=50, fmax=14000 Hz |
| 4 | Log-scale | `power_to_db` conversion |
| 5 | Normalise | Per-clip min-max normalisation to `[0, 1]` |
| 6 | Tensorise | Shape `(1, 128, 313)` — single channel, ready for CNN input |
| 7 | SpecAugment | Training only: frequency masking (≤24 bins) + time masking (≤64 frames) |

### Why these constants?

- **32000 Hz**: confirmed sample rate of all BirdCLEF 2026 audio files
- **5 seconds**: standard BirdCLEF clip length, covers most complete bird call sequences
- **128 mel bins**: standard resolution balancing frequency detail and compute cost
- **313 time frames**: derived from `sr=32000, hop=512, n_fft=1024, duration=5s`
- **fmin=50 Hz**: removes low-frequency wind and handling noise
- **fmax=14000 Hz**: covers the relevant acoustic range for bird vocalizations

---

## Label Encoding

Labels are encoded as **multi-hot binary vectors** of shape `(K,)` where `K = 206`
(all species in `train_audio`). The `MultiLabelBinarizer` provides a fixed, sorted
mapping from species code strings to vector indices.

- `train_audio` samples: exactly one active label per clip (label sum = 1)
- `train_soundscapes` samples: one or more active labels per segment (label sum ≥ 1)
- Soundscape species not in `train_audio` are silently ignored

---

## Data Split Strategy

Three-way stratified split by `primary_label` (PROBLEMSETTING.md §Evaluation Protocol):

```
train.csv (35,549 recordings, 206 species)
         │
         ├── ~70% → Training set
         │         + all labeled train_soundscapes segments
         │         + SpecAugment augmentation enabled
         │
         ├── ~15% → Validation set (stratified, no augmentation)
         │         used for: per-epoch metrics, checkpoint selection,
         │         per-class threshold tuning, HP selection
         │
         └── ~15% → Test set (stratified, no augmentation)
                   used only for final reporting via evaluate.py --split test
                   thresholds are NEVER tuned on the test set
```

Singleton-class recordings (only 1 example) are moved to train only.
All models use the same `--seed`, `--val_size`, `--test_size` defaults for comparable results.

### Optional species filter (`min_recordings`)

For HP search, pass `min_recordings=200` to restrict to K=69 species
(PROBLEMSETTING §Scope fallback). Reduces dataset to ~26% of full size.

---

## SpecAugment

Applied only during training:

- **FrequencyMasking** (`freq_mask_param=24`): randomly zeros up to 24 consecutive mel bins
- **TimeMasking** (`time_mask_param=64`): randomly zeros up to 64 consecutive time frames

---

## Public API

```python
from data.preprocessing.data_pipeline import build_dataloaders

train_loader, val_loader, test_loader, mlb = build_dataloaders(
    metadata_csv    = "path/to/train.csv",
    audio_dir       = "path/to/train_audio/",
    soundscapes_dir = "path/to/train_soundscapes/",
    soundscapes_csv = "path/to/train_soundscapes_labels.csv",
    val_size        = 0.15,
    test_size       = 0.15,
    batch_size      = 32,
    num_workers     = 4,
    random_state    = 42,
    min_recordings  = None,   # set to 200 for K=69 HP search subset
)
```

### Returns

| Object | Type | Description |
| ------ | ---- | ----------- |
| `train_loader` | `DataLoader` | Batches with SpecAugment |
| `val_loader` | `DataLoader` | Batches without augmentation |
| `test_loader` | `DataLoader` | Batches without augmentation — use only for final evaluation |
| `mlb` | `MultiLabelBinarizer` | Fitted label encoder — keep for inference |

### Batch format

```python
specs,  # torch.Tensor — shape (batch_size, 1, 128, 313)
labels  # torch.Tensor — shape (batch_size, K), dtype float32, values in {0, 1}
```

---

## Sanity Check

Open `data/preprocessing/preprocessing_check.ipynb` in Jupyter and run all cells.
It plots three spectrograms: training sample (SpecAugment masks expected), validation
sample (no masks), and a multi-label soundscape sample. Set `$DATA_ROOT` to the
BirdCLEF 2026 data directory before running, or edit `DATA_ROOT` directly in the notebook.
