"""Soundscape-domain validation loader.

Holds out a fraction of train_soundscapes recordings (split by filename, not
segment) as a secondary eval set. Measures domain generalisation: the gap
between clean train_audio val AUC and soundscape AUC is the real optimisation
target for the competition test set.

Split is by *file* to prevent consecutive-segment leakage: all 5s segments
from the same recording go entirely to train or entirely to val.

By default (report track) the val loader is logged each epoch only.
With ``train.py --checkpoint_metric sc_auc`` (Kaggle track) the same loader
may drive checkpoint selection.
"""
from __future__ import annotations

from collections import defaultdict

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


def split_soundscape_val_files(
    soundscapes_csv: str,
    soundscapes_dir: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> frozenset[str] | None:
    """File-level holdout paths for soundscape eval (no MLB required).

    Returns ``None`` when val_size <= 0, no samples exist, or too few files.
    """
    if val_size <= 0.0:
        return None

    all_samples = build_samples_from_soundscapes(soundscapes_csv, soundscapes_dir)
    if not all_samples:
        return None

    by_file: dict[str, list] = defaultdict(list)
    for s in all_samples:
        by_file[s["file_path"]].append(s)

    files = sorted(by_file.keys())
    if len(files) < 2:
        print("soundscape_eval: too few files to split — skipping soundscape val")
        return None

    train_files, val_files = train_test_split(
        files, test_size=val_size, random_state=random_state
    )
    n_train_seg = sum(len(by_file[f]) for f in train_files)
    n_val_seg = sum(len(by_file[f]) for f in val_files)
    print(
        f"soundscape split — files: {len(train_files)} train / {len(val_files)} val  "
        f"| segments: {n_train_seg} train / {n_val_seg} val"
    )
    return frozenset(val_files)


def build_soundscape_val_loader(
    soundscapes_csv: str,
    soundscapes_dir: str,
    mlb,
    val_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    random_state: int = 42,
    val_files: frozenset[str] | set[str] | None = None,
) -> tuple[DataLoader | None, frozenset[str] | None]:
    """Build soundscape val DataLoader from a file-level holdout.

    Returns ``(val_loader, val_files)``. Both are ``None`` when soundscape val
    is disabled or no files exist.

    Pass ``val_files`` from :func:`split_soundscape_val_files` (called before
    ``build_dataloaders``) so training excludes the same holdout recordings.
    """
    if val_files is None:
        val_files = split_soundscape_val_files(
            soundscapes_csv, soundscapes_dir, val_size, random_state
        )
    if not val_files:
        return None, None

    all_samples = build_samples_from_soundscapes(soundscapes_csv, soundscapes_dir)
    val_samples = [s for s in all_samples if s["file_path"] in val_files]
    if not val_samples:
        return None, None

    val_ds = BirdCLEFDataset(val_samples, mlb, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return val_loader, val_files
