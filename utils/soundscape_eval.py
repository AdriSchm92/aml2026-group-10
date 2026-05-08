"""Soundscape-domain validation loader.

Holds out a fraction of train_soundscapes recordings (split by filename, not
segment) as a secondary eval set. Measures domain generalisation: the gap
between clean train_audio val AUC and soundscape AUC is the real optimisation
target for the competition test set.

Split is by *file* to prevent consecutive-segment leakage: all 5s segments
from the same recording go entirely to train or entirely to val.
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import (  # noqa: E402
    BirdCLEFDataset,
    build_samples_from_soundscapes,
)


def build_soundscape_val_loader(
    soundscapes_csv: str,
    soundscapes_dir: str,
    mlb,
    val_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    random_state: int = 42,
) -> tuple[DataLoader | None, DataLoader | None]:
    """Split soundscape recordings into train/val by filename.

    Returns (soundscape_train_loader, soundscape_val_loader).
    Both are None if no soundscape files exist or val_size == 0.
    The val loader is evaluation-only; never use for checkpointing.
    """
    if val_size <= 0.0:
        return None, None

    all_samples = build_samples_from_soundscapes(soundscapes_csv, soundscapes_dir)
    if not all_samples:
        return None, None

    # group segments by source file — split files, not segments
    from collections import defaultdict
    by_file: dict[str, list] = defaultdict(list)
    for s in all_samples:
        by_file[s["file_path"]].append(s)

    files = sorted(by_file.keys())
    if len(files) < 2:
        print("soundscape_eval: too few files to split — skipping soundscape val")
        return None, None

    train_files, val_files = train_test_split(
        files, test_size=val_size, random_state=random_state
    )

    train_samples = [s for f in train_files for s in by_file[f]]
    val_samples   = [s for f in val_files   for s in by_file[f]]

    print(
        f"soundscape split — files: {len(train_files)} train / {len(val_files)} val  "
        f"| segments: {len(train_samples)} train / {len(val_samples)} val"
    )

    train_ds = BirdCLEFDataset(train_samples, mlb, augment=True)
    val_ds   = BirdCLEFDataset(val_samples,   mlb, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
