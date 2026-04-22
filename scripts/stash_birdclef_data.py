"""Copy the BirdCLEF / Kaggle competition tree into a repo-local stash for fast I/O.

  python scripts/stash_birdclef_data.py --source /path/to/birdclef-2026

Default destination: <repo>/birdclef_stash/ or BIRDCLEF_STASH_DIR. Uses ``rsync -a`` when
available (incremental). Pass ``--with-delete`` to mirror and remove files missing at
the source. Without ``rsync``, falls back to ``shutil.copytree`` (slower, not resumable).

Afterwards run ``train.py`` without ``DATA_ROOT`` to pick up the stash (see train.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STASH = REPO_ROOT / "birdclef_stash"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy competition data into a local birdclef_stash/ directory"
    )
    p.add_argument(
        "--source",
        required=True,
        help="Existing competition root (contains train.csv, train_audio/, …)",
    )
    p.add_argument(
        "--dest",
        default=None,
        help="Stash path (default: BIRDCLEF_STASH_DIR or <repo>/birdclef_stash/)",
    )
    p.add_argument(
        "--with-delete",
        action="store_true",
        help="Pass rsync --delete: dest will exactly mirror source (removes extra files in dest).",
    )
    p.add_argument(
        "--no-rsync",
        action="store_true",
        help="Use shutil only (e.g. when rsync missing); slow, not resumable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    if not (source / "train.csv").is_file():
        print(f"error: {source}/train.csv not found. Is this the competition data root?")
        raise SystemExit(1)

    dest = Path(
        args.dest
        or (os.environ.get("BIRDCLEF_STASH_DIR") or "").strip()
        or DEFAULT_STASH
    ).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"source: {source}")
    print(f"dest:   {dest}")

    if shutil.which("rsync") and not args.no_rsync:
        cmd: list[str] = [
            "rsync",
            "-a",
            f"{str(source)}/",
            f"{str(dest)}/",
        ]
        if args.with_delete:
            cmd.insert(1, "--delete")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        if args.with_delete and dest.is_dir() and any(dest.iterdir()):
            print("removing dest (--with-delete, no rsync)")
            shutil.rmtree(dest)
        print(
            "Warning: no rsync or --no-rsync; using shutil.copytree "
            "(slower, not resumable on interrupt)."
        )
        shutil.copytree(source, dest, dirs_exist_ok=True)

    if not (dest / "train.csv").is_file():
        print("error: copy finished but train.csv missing at dest")
        raise SystemExit(1)

    print("Done. Run: python train.py  (unset DATA_ROOT) to use this stash, or set "
          "BIRDCLEF_STASH_DIR to the path above.")


if __name__ == "__main__":
    main()
