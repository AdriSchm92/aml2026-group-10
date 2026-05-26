"""Generate submission.csv for the BirdCLEF 2026 Kaggle competition.

Reads test_soundscapes/, runs the trained model on each 5-second window, and
writes submission.csv in the format expected by the competition:

    row_id                   species1  species2  ...
    soundscape_29201_5       0.12      0.03      ...
    soundscape_29201_10      0.08      0.27      ...
    ...

row_id = {soundscape_filename_stem}_{end_second_of_window}

Usage — local test (points at the competition data on disk):
    python scripts/kaggle_submit.py \
        --checkpoint /path/to/best_pretrained_transformer_v2.pt \
        --data_root /path/to/birdclef-2026 \
        --output submission.csv

Usage — inside a Kaggle Notebook (paths auto-resolved):
    import subprocess
    subprocess.run(["python", "scripts/kaggle_submit.py",
                    "--checkpoint", "/kaggle/input/my-birdclef-model/best_pretrained_transformer_v2.pt",
                    "--data_root",  "/kaggle/input/birdclef-2026"])

The script supports ensembling: pass --checkpoint multiple times.
Probabilities are averaged across checkpoints before writing.

    python scripts/kaggle_submit.py \
        --checkpoint /path/to/best_cnn_baseline.pt \
        --checkpoint /path/to/best_pretrained_transformer_v2.pt \
        --data_root /path/to/birdclef-2026 \
        --output submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import CLIP_DURATION, SAMPLE_RATE  # noqa: E402
from models.registry import load_model  # noqa: E402
from utils.inference import predict_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BirdCLEF 2026 Kaggle submission")
    p.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        required=True,
        metavar="PATH",
        help="Path to .pt checkpoint. Repeat for ensemble (probabilities averaged).",
    )
    p.add_argument(
        "--data_root",
        default=None,
        help="Competition data root. Defaults to /kaggle/input/birdclef-2026 "
             "inside Kaggle, or $BIRDCLEF_DATA_ROOT locally.",
    )
    p.add_argument(
        "--output",
        default="submission.csv",
        help="Output path for submission CSV (default: submission.csv).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for per-file chunk inference.",
    )
    return p.parse_args()


def resolve_data_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = __import__("os").environ.get("BIRDCLEF_DATA_ROOT")
    if env:
        return Path(env)
    kaggle_default = Path("/kaggle/input/birdclef-2026")
    if kaggle_default.exists():
        return kaggle_default
    raise FileNotFoundError(
        "Cannot find competition data. Pass --data_root or set $BIRDCLEF_DATA_ROOT."
    )


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load a checkpoint and return (model, class_list)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt["num_classes"]
    model_kwargs = ckpt.get("model_kwargs", {}) or {}
    model = load_model(ckpt["model_name"], num_classes, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    class_list = ckpt.get("classes")
    print(
        f"  Loaded {ckpt_path.name} — model={ckpt['model_name']}, "
        f"K={num_classes}, epoch={ckpt.get('epoch', '?')}, "
        f"val_auc={ckpt.get('val_auc', float('nan')):.4f}"
    )
    return model, class_list, num_classes


def get_window_end_seconds(file_path: Path) -> list[int]:
    """Return the end-second for every 5s window in the audio file."""
    import librosa
    duration = librosa.get_duration(path=str(file_path))
    n_windows = max(1, int(np.ceil(duration / CLIP_DURATION)))
    return [(i + 1) * CLIP_DURATION for i in range(n_windows)]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = resolve_data_root(args.data_root)
    test_dir = data_root / "test_soundscapes"
    sample_sub_path = data_root / "sample_submission.csv"

    if not test_dir.exists():
        raise FileNotFoundError(f"test_soundscapes/ not found at {test_dir}")

    # ── 1. Load all checkpoints ───────────────────────────────────────────────
    print(f"\nLoading {len(args.checkpoints)} checkpoint(s):")
    models = []
    shared_classes: list[str] | None = None
    for cp in args.checkpoints:
        model, classes, _ = load_checkpoint(Path(cp), device)
        models.append(model)
        if shared_classes is None:
            shared_classes = classes
        elif classes is not None and classes != shared_classes:
            print(
                "  WARNING: class list mismatch between checkpoints. "
                "Using class list from first checkpoint."
            )

    # ── 2. Determine species column order ────────────────────────────────────
    # Prefer sample_submission.csv so our column order matches Kaggle exactly.
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        species_cols = [c for c in sample_sub.columns if c != "row_id"]
        print(f"\nSpecies columns from sample_submission.csv: {len(species_cols)} classes")
    elif shared_classes is not None:
        species_cols = shared_classes
        print(f"\nSpecies columns from checkpoint class list: {len(species_cols)} classes")
    else:
        raise RuntimeError(
            "Cannot determine species order: no sample_submission.csv and no class "
            "list in checkpoint. Re-save checkpoint with train.py (it saves 'classes')."
        )

    species_to_idx = {s: i for i, s in enumerate(species_cols)}

    # ── 3. Discover test files ────────────────────────────────────────────────
    test_files = sorted(test_dir.glob("*.ogg"))
    if not test_files:
        test_files = sorted(test_dir.glob("*.flac"))
    if not test_files:
        raise FileNotFoundError(f"No .ogg/.flac files found in {test_dir}")
    print(f"Test soundscapes found: {len(test_files)}")

    # ── 4. Run inference ──────────────────────────────────────────────────────
    rows: list[dict] = []
    for file_idx, file_path in enumerate(test_files):
        stem = file_path.stem  # e.g. "soundscape_29201"

        # Ensemble: average probabilities across all checkpoints.
        # predict_file returns (clip_probs, window_probs) — we need window_probs.
        ensemble_window_probs: np.ndarray | None = None
        for model in models:
            _, window_probs = predict_file(
                model, str(file_path), device, chunk_batch_size=args.batch_size
            )
            if ensemble_window_probs is None:
                ensemble_window_probs = window_probs
            else:
                ensemble_window_probs += window_probs
        ensemble_window_probs /= len(models)  # type: ignore[operator]

        n_windows = ensemble_window_probs.shape[0]  # type: ignore[union-attr]
        for w_idx in range(n_windows):
            end_sec = (w_idx + 1) * CLIP_DURATION
            row_id = f"{stem}_{end_sec}"

            # Map model output indices to competition species order.
            # If model K == len(species_cols) and shared_classes matches, this
            # is a direct copy. Otherwise we map by name.
            if shared_classes is not None and len(shared_classes) == len(species_cols):
                probs = ensemble_window_probs[w_idx]
            else:
                # Build a zero-initialised row and fill known species.
                probs = np.zeros(len(species_cols), dtype=np.float32)
                if shared_classes is not None:
                    for i, sp in enumerate(shared_classes):
                        if sp in species_to_idx:
                            probs[species_to_idx[sp]] = ensemble_window_probs[w_idx, i]  # type: ignore[index]

            row = {"row_id": row_id}
            row.update(dict(zip(species_cols, probs.tolist())))
            rows.append(row)

        if (file_idx + 1) % 10 == 0 or (file_idx + 1) == len(test_files):
            print(f"  [{file_idx + 1}/{len(test_files)}] {stem} — {n_windows} windows")

    # ── 5. Write submission ───────────────────────────────────────────────────
    sub_df = pd.DataFrame(rows, columns=["row_id"] + species_cols)

    # If sample_submission.csv exists, align row order to it.
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        sub_df = sample_sub[["row_id"]].merge(sub_df, on="row_id", how="left")
        # Fill any rows that had no prediction with 0.
        sub_df[species_cols] = sub_df[species_cols].fillna(0.0)

    sub_df.to_csv(args.output, index=False)
    print(f"\nSubmission written to {args.output}  ({len(sub_df)} rows)")


if __name__ == "__main__":
    main()
