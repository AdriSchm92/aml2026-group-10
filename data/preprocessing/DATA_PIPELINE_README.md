# data_pipeline.py — Data Pipeline for BirdCLEF 2026

## Overview

This module implements the full data pipeline for the BirdCLEF 2026 bird species
identification task. It handles loading, preprocessing, augmentation, and batching
from two complementary data sources into PyTorch DataLoaders ready for model training.

---

## Data Sources

### 1. `train_audio/` — Clean single-species recordings
- Sourced from Xeno-canto and iNaturalist, uploaded by birdwatchers
- Each file contains one primary species, recorded in relatively clean conditions
- Used for both training and validation
- Labels: single species per recording (`primary_label` column in `train.csv`)

### 2. `train_soundscapes/` — Real-world passive acoustic recordings
- Recorded from the same field locations and equipment as the hidden test set
- Contain multiple species vocalizing simultaneously over natural background noise
- Used for training only (not validation) to close the domain gap between train and test
- Labels: expert-annotated, semicolon-separated species lists per 5-second segment
  (`train_soundscapes_labels.csv`)

---

## Preprocessing Pipeline

Each audio clip goes through the following steps in `__getitem__`:

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Load audio | 5-second window at `offset` seconds, resampled to 32000 Hz mono |
| 2 | Zero-pad | Short clips (< 5s) padded with silence on the right |
| 3 | Mel-spectrogram | Shape `(128, 313)`, hop=512, n_fft=1024, fmin=50, fmax=14000 Hz |
| 4 | Log-scale | `power_to_db` conversion |
| 5 | Normalise | Per-clip min-max normalisation to `[0, 1]` |
| 6 | Tensorise | Shape `(1, 128, 313)` — single channel, ready for CNN input |
| 7 | SpecAugment | Training only: random frequency masking (≤24 bins) + time masking (≤64 frames) |

### Why these constants?
- **32000 Hz**: confirmed sample rate of all BirdCLEF 2026 audio files
- **5 seconds**: standard BirdCLEF clip length, covers most complete bird call sequences
- **128 mel bins**: standard resolution balancing frequency detail and compute cost
- **313 time frames**: derived from `sr=32000, hop=512, n_fft=1024, duration=5s` with librosa's default center padding
- **fmin=50 Hz**: removes low-frequency wind and handling noise
- **fmax=14000 Hz**: covers the relevant acoustic range for bird vocalizations

---

## Label Encoding

Labels are encoded as **multi-hot binary vectors** of shape `(K,)` where `K = 206`
(all species in `train_audio`). The `MultiLabelBinarizer` provides a fixed,
sorted mapping from species code strings (e.g. `"rufgna3"`) to vector indices,
ensuring consistency across train, validation, and test sets.

- `train_audio` samples: exactly one active label per clip (label sum = 1)
- `train_soundscapes` samples: one or more active labels per segment (label sum ≥ 1)
- Soundscape species not present in `train_audio` are silently ignored

---

## Data Split Strategy

```
train.csv (35,549 recordings, 206 species)
         │
         ├── 85% → Training set
         │         + all labeled train_soundscapes segments
         │         + SpecAugment augmentation enabled
         │
         └── 15% → Validation set (stratified by species)
                   no soundscapes, no augmentation
                   used for hyperparameter selection and model comparison
```

The validation set is drawn **only from `train_audio`** to keep evaluation clean,
consistent, and comparable across all models. Soundscape segments are training-only
since their multi-label nature and noisier annotations would complicate metric interpretation.

---

## SpecAugment

Applied only during training to improve robustness to real-world acoustic conditions:

- **FrequencyMasking** (`freq_mask_param=24`): randomly zeros up to 24 consecutive
  mel bins, simulating frequency-band corruption from noise or interference
- **TimeMasking** (`time_mask_param=64`): randomly zeros up to 64 consecutive time
  frames, simulating brief silences or call interruptions

These values are treated as hyperparameters and may be tuned in a later phase.

---

## Public API

```python
from data_pipeline import build_dataloaders

train_loader, val_loader, mlb = build_dataloaders(
    metadata_csv    = "path/to/train.csv",
    audio_dir       = "path/to/train_audio/",
    soundscapes_dir = "path/to/train_soundscapes/",
    soundscapes_csv = "path/to/train_soundscapes_labels.csv",
    val_size        = 0.15,      # fraction of train_audio held out for validation
    batch_size      = 32,        # adjust based on available GPU memory
    num_workers     = 4,         # parallel workers for data loading
    random_state    = 42,        # for reproducibility
)
```

### Returns
| Object | Type | Description |
|--------|------|-------------|
| `train_loader` | `DataLoader` | Batches of `(specs, labels)` with augmentation |
| `val_loader` | `DataLoader` | Batches of `(specs, labels)` without augmentation |
| `mlb` | `MultiLabelBinarizer` | Fitted label encoder — keep this for inference |

### Batch format
```python
specs,  # torch.Tensor — shape (batch_size, 1, 128, 313)
labels  # torch.Tensor — shape (batch_size, 206), dtype float32, values in {0, 1}
```

---

## Sanity Check

Run the script directly to verify the full pipeline visually:

```bash
python data_pipeline.py
```

This plots three spectrograms side by side:
1. **Training sample** — should show SpecAugment masks (dark horizontal/vertical blocks)
2. **Validation sample** — should be clean with no masks
3. **Multi-label soundscape** — should show label sum > 1 in the title
