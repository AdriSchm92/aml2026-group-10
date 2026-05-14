"""RF baseline training script for BirdCLEF 2026.

MFCC (mean+std over time) features + OneVsRestClassifier(RandomForest).
Standalone sklearn pipeline — does not use the PyTorch train.py harness.

Feature extraction is parallelised with joblib and cached per (split, n_mfcc)
under .cache/birdclef/ to avoid re-extraction across HP grid runs.

Usage:
    # Single run with default HPs
    python train_rf.py

    # Full HP grid (3 × n_mfcc × 2 × n_estimators × 2 × max_depth = 12 combos)
    python train_rf.py --tune

    # Renku
    python train_rf.py --tune --n_jobs 10 \\
        --output_dir /home/renku/work/kaggle-data/aml2026-group10-runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import (  # noqa: E402
    build_samples_from_soundscapes,
    build_samples_from_train_audio,
)
from models.rf_baseline import build_model, extract_features  # noqa: E402
from train import resolve_data_root, resolve_output_dir  # noqa: E402
from utils.metrics import macro_f1_tuned, macro_roc_auc  # noqa: E402

CACHE_ROOT = REPO_ROOT / ".cache" / "birdclef"

HP_GRID = [
    {"n_mfcc": n_mfcc, "n_estimators": n_est, "max_depth": max_d}
    for n_mfcc in [20, 40, 80]
    for n_est in [100, 300]
    for max_d in [None, 20]
]


# ── Feature helpers ───────────────────────────────────────────────────────────

def _cache_path(split: str, n_mfcc: int) -> Path:
    return CACHE_ROOT / f"rf_features_{split}_mfcc{n_mfcc}.npz"


def extract_or_load(samples: list[dict], n_mfcc: int, n_jobs: int, path: Path) -> np.ndarray:
    if path.is_file():
        print(f"  cache hit: {path.name}")
        return np.load(path)["X"]
    print(f"  extracting {len(samples)} samples  n_mfcc={n_mfcc}  n_jobs={n_jobs} ...")
    t0 = time.time()
    X = np.array(
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(extract_features)(s["file_path"], s["offset"], n_mfcc)
            for s in samples
        ),
        dtype=np.float32,
    )
    print(f"  done {time.time() - t0:.1f}s  shape={X.shape}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), X=X)
    return X


def samples_to_labels(samples: list[dict], mlb: MultiLabelBinarizer) -> np.ndarray:
    known = set(mlb.classes_)
    filtered = [[l for l in s["labels"] if l in known] for s in samples]
    return mlb.transform(filtered).astype(np.float32)


# ── Soundscape val features ───────────────────────────────────────────────────

def build_soundscape_val_features(
    soundscapes_csv: str,
    soundscapes_dir: str,
    mlb: MultiLabelBinarizer,
    val_size: float,
    n_mfcc: int,
    n_jobs: int,
    random_state: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if val_size <= 0.0:
        return None, None
    all_samples = build_samples_from_soundscapes(soundscapes_csv, soundscapes_dir)
    if not all_samples:
        return None, None
    by_file: dict[str, list] = defaultdict(list)
    for s in all_samples:
        by_file[s["file_path"]].append(s)
    files = sorted(by_file.keys())
    if len(files) < 2:
        print("soundscape_val: too few files to split — skipping")
        return None, None
    _, val_files = train_test_split(files, test_size=val_size, random_state=random_state)
    val_samples = [s for f in val_files for s in by_file[f]]
    print(f"soundscape val: {len(val_samples)} segments from {len(val_files)} files")
    X_sc = extract_or_load(val_samples, n_mfcc, n_jobs, _cache_path("sc_val", n_mfcc))
    y_sc = samples_to_labels(val_samples, mlb)
    return X_sc, y_sc


# ── Single HP run ─────────────────────────────────────────────────────────────

def run_single(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_sc: np.ndarray | None,
    y_sc: np.ndarray | None,
    n_estimators: int,
    max_depth: int | None,
    n_mfcc: int,
    n_jobs: int,
    random_state: int,
) -> dict:
    model = build_model(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=n_jobs, random_state=random_state,
    )
    print(f"  fitting: n_mfcc={n_mfcc}  n_estimators={n_estimators}  max_depth={max_depth}")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"  fit done {fit_time:.1f}s")

    y_score = model.predict_proba(X_val)
    auc, _ = macro_roc_auc(y_val, y_score)
    f1, thresholds = macro_f1_tuned(y_val, y_score)
    print(f"  val_auc={auc:.4f}  val_f1={f1:.4f}")

    sc_auc, sc_f1 = float("nan"), float("nan")
    if X_sc is not None:
        sc_score = model.predict_proba(X_sc)
        sc_auc, _ = macro_roc_auc(y_sc, sc_score)
        sc_f1, _ = macro_f1_tuned(y_sc, sc_score)
        print(f"  sc_auc={sc_auc:.4f}  sc_f1={sc_f1:.4f}")

    return {
        "model": model,
        "thresholds": thresholds,
        "metrics": {
            "n_mfcc": n_mfcc,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "val_auc": auc,
            "val_f1": f1,
            "sc_auc": sc_auc,
            "sc_f1": sc_f1,
            "fit_time_s": fit_time,
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF baseline for BirdCLEF 2026")
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_mfcc", type=int, default=40,
                   help="MFCC feature dimension. Ignored when --tune is set.")
    p.add_argument("--n_estimators", type=int, default=100,
                   help="RF trees per class. Ignored when --tune is set.")
    p.add_argument("--max_depth", type=int, default=None,
                   help="RF max depth (None = unlimited). Ignored when --tune is set.")
    p.add_argument("--n_jobs", type=int, default=10,
                   help="Parallel workers for feature extraction + RF tree building.")
    p.add_argument("--soundscape_val_size", type=float, default=0.2)
    p.add_argument("--tune", action="store_true",
                   help="Run full 12-combo HP grid. Ignores --n_mfcc / --n_estimators / --max_depth.")
    p.add_argument("--tag", default=None)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root).resolve()
    output_dir = resolve_output_dir(args.output_dir, data_root)
    print(f"DATA_ROOT  : {data_root}")
    print(f"OUTPUT_DIR : {output_dir}")
    print(f"ARGS       : {vars(args)}")

    # ── Same stratified split as train.py ────────────────────────────────────
    df = pd.read_csv(data_root / "train.csv")
    counts = df["primary_label"].value_counts()
    singletons = counts[counts < 2].index
    df_main = df[~df["primary_label"].isin(singletons)]
    df_sing = df[df["primary_label"].isin(singletons)]
    train_df, val_df = train_test_split(
        df_main, test_size=args.val_size,
        stratify=df_main["primary_label"], random_state=args.seed,
    )
    if len(df_sing):
        train_df = pd.concat([train_df, df_sing], ignore_index=True)

    mlb = MultiLabelBinarizer()
    mlb.fit([[s] for s in sorted(df["primary_label"].unique())])
    print(f"K={len(mlb.classes_)}  train_recordings={len(train_df)}  val_recordings={len(val_df)}")

    train_samples = build_samples_from_train_audio(
        train_df, str(data_root / "train_audio"), split_label="train"
    )
    val_samples = build_samples_from_train_audio(
        val_df, str(data_root / "train_audio"), split_label="val"
    )
    print(f"train_chunks={len(train_samples)}  val_chunks={len(val_samples)}")

    # ── HP grid ───────────────────────────────────────────────────────────────
    grid = HP_GRID if args.tune else [
        {"n_mfcc": args.n_mfcc, "n_estimators": args.n_estimators, "max_depth": args.max_depth}
    ]
    tag = f"_{args.tag}" if args.tag else ""
    metrics_path = output_dir / f"metrics_rf_baseline{tag}.jsonl"
    metrics_path.write_text("")

    best_auc = -1.0
    best_result: dict | None = None

    for hp in grid:
        n_mfcc = hp["n_mfcc"]
        print(f"\n── {hp} ──────────────────────────────────────────────────────")

        X_train = extract_or_load(train_samples, n_mfcc, args.n_jobs,
                                  _cache_path("train", n_mfcc))
        X_val   = extract_or_load(val_samples,   n_mfcc, args.n_jobs,
                                  _cache_path("val",   n_mfcc))
        y_train = samples_to_labels(train_samples, mlb)
        y_val   = samples_to_labels(val_samples,   mlb)

        X_sc, y_sc = build_soundscape_val_features(
            str(data_root / "train_soundscapes_labels.csv"),
            str(data_root / "train_soundscapes"),
            mlb, args.soundscape_val_size, n_mfcc, args.n_jobs, args.seed,
        )

        result = run_single(
            X_train, y_train, X_val, y_val, X_sc, y_sc,
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            n_mfcc=n_mfcc,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

        with metrics_path.open("a") as f:
            f.write(json.dumps(result["metrics"]) + "\n")

        if result["metrics"]["val_auc"] > best_auc:
            best_auc = result["metrics"]["val_auc"]
            best_result = result
            print(f"  -> new best val_auc={best_auc:.4f}")

    # ── Save best checkpoint ──────────────────────────────────────────────────
    assert best_result is not None
    ckpt_path = output_dir / f"best_rf_baseline{tag}.joblib"
    joblib.dump(
        {
            "model": best_result["model"],
            "mlb": mlb,
            "thresholds": best_result["thresholds"],
            "val_auc": best_auc,
            "hp": best_result["metrics"],
        },
        ckpt_path,
    )
    print(f"\nDone. Best val AUC : {best_auc:.4f}")
    print(f"Checkpoint         : {ckpt_path}")
    print(f"Metrics            : {metrics_path}")


if __name__ == "__main__":
    main()
