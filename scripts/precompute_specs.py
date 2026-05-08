"""Pre-compute mel spectrograms and save to the spec cache before training.

Run this ONCE to populate the cache on persistent storage (SwitchDrive).
After it finishes, training epochs load ~1ms .npy files instead of decoding
OGG files from the network mount, keeping the GPU busy and the Renku session alive.

Usage:
    export BIRDCLEF_SPEC_CACHE=/home/renku/work/kaggle-data/birdclef_specs
    export DATA_ROOT=/home/renku/work/kaggle-data/birdclef-2026
    python scripts/precompute_specs.py

    # K=69 subset only (faster, enough for HP search)
    python scripts/precompute_specs.py --min_recordings 200

    # Use more parallel workers to go faster
    python scripts/precompute_specs.py --num_workers 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import (  # noqa: E402
    _save_spec_cache,
    _spec_cache_key,
    _load_spec_cache,
    build_samples_from_train_audio,
    build_samples_from_soundscapes,
    BirdCLEFDataset,
)
from train import resolve_data_root  # noqa: E402

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute spectrogram cache")
    p.add_argument("--data_root", default=None)
    p.add_argument("--spec_cache_dir", default=os.environ.get("BIRDCLEF_SPEC_CACHE"),
                   help="Cache directory. Defaults to BIRDCLEF_SPEC_CACHE env var.")
    p.add_argument("--min_recordings", type=int, default=None,
                   help="Only cache species with >= N recordings (e.g. 200 for K=69 subset).")
    p.add_argument("--num_workers", type=int, default=8,
                   help="Parallel DataLoader workers for building the cache.")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.spec_cache_dir:
        print("ERROR: set --spec_cache_dir or BIRDCLEF_SPEC_CACHE env var.")
        print("  e.g.  export BIRDCLEF_SPEC_CACHE=/home/renku/work/kaggle-data/birdclef_specs")
        sys.exit(1)

    data_root = resolve_data_root(args.data_root)
    print(f"Data root  : {data_root}")
    print(f"Cache dir  : {args.spec_cache_dir}")

    df = pd.read_csv(data_root / "train.csv")
    if args.min_recordings:
        counts = df["primary_label"].value_counts()
        keep = counts[counts >= args.min_recordings].index
        df = df[df["primary_label"].isin(keep)].reset_index(drop=True)
        print(f"Subset     : {len(df)} recordings, {df['primary_label'].nunique()} species")

    mlb = MultiLabelBinarizer()
    mlb.fit([[s] for s in sorted(df["primary_label"].unique())])

    audio_samples = build_samples_from_train_audio(df, str(data_root / "train_audio"))
    soundscape_samples = build_samples_from_soundscapes(
        str(data_root / "train_soundscapes_labels.csv"),
        str(data_root / "train_soundscapes"),
    )
    all_samples = audio_samples + soundscape_samples
    print(f"Total samples to cache: {len(all_samples)}")

    # Count already cached
    cache_dir = args.spec_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    already = sum(
        1 for s in all_samples
        if os.path.exists(os.path.join(cache_dir, _spec_cache_key(s["file_path"], s["offset"]) + ".npy"))
    )
    remaining = len(all_samples) - already
    print(f"Already cached: {already}  |  Remaining: {remaining}")
    if remaining == 0:
        print("Cache is complete. Nothing to do.")
        return

    # Use BirdCLEFDataset with augment=False to iterate and populate cache.
    # The dataset's __getitem__ writes to cache automatically on cache miss.
    dataset = BirdCLEFDataset(all_samples, mlb, augment=False, spec_cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    t0 = time.time()
    n_batches = len(loader)
    for i, _ in enumerate(loader, 1):
        if i % 50 == 0 or i == n_batches:
            elapsed = time.time() - t0
            rate = (i * args.batch_size) / elapsed
            remaining_s = (n_batches - i) * args.batch_size / max(rate, 1)
            print(
                f"  [{i}/{n_batches}]  "
                f"{i * args.batch_size}/{len(all_samples)} samples  "
                f"{rate:.0f} samples/s  "
                f"ETA {remaining_s/60:.1f} min"
            )

    total = time.time() - t0
    print(f"\nDone. {len(all_samples)} specs cached in {total/60:.1f} min -> {cache_dir}")


if __name__ == "__main__":
    main()
