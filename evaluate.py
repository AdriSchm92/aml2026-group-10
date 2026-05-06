"""Standalone evaluation on val or test split for a saved checkpoint.

Loads the model via models.registry (same as train.py) and reuses the
DataLoaders from data.preprocessing.data_pipeline.

Split behaviour (PROBLEMSETTING.md §Evaluation Protocol):
  --split val  (default) — metrics on the validation set; thresholds tuned on
               the same split and printed. Behaviour unchanged from original.
  --split test — threshold source is the checkpoint (tuned on val at save time).
               Applies those thresholds to the held-out test set without any
               test-set tuning (which would leak). Emits test_results_<model>.json.

All models must use the same seed/val_size/test_size for results to be
directly comparable (PROBLEMSETTING §Evaluation Protocol).

Usage:
  python evaluate.py --checkpoint runs/best_cnn_baseline.pt
  python evaluate.py --model cnn_baseline
  python evaluate.py --model cnn_transformer --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import build_dataloaders  # noqa: E402
from models.registry import load_model  # noqa: E402
from train import resolve_data_root, resolve_output_dir  # noqa: E402
from utils.inference import predict_val_probs  # noqa: E402
from utils.metrics import macro_f1_tuned, macro_roc_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt from training. If omitted, uses OUTPUT_DIR/best_<model>.pt "
        "(same resolution as train: --output_dir, TRAINING_OUTPUT_DIR, Renku default).",
    )
    p.add_argument(
        "--model",
        default="cnn_baseline",
        help="Checkpoint stem when --checkpoint is omitted (best_<model>[_tag].pt).",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Same as train.py --tag when resolving default checkpoint path.",
    )
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate. 'val' tunes thresholds on the eval set "
             "(exploratory). 'test' uses val-tuned thresholds saved in the "
             "checkpoint (final reporting — no leakage).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override artifact directory (same as train.py --output_dir).",
    )
    p.add_argument("--data_root", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_size", type=float, default=0.15,
                   help="Must match the value used during training for identical splits.")
    p.add_argument("--test_size", type=float, default=0.15,
                   help="Must match the value used during training for identical splits.")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the value used during training for identical splits.")
    p.add_argument("--save_per_class", default=None,
                   help="Optional path to write per-class AUC JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    data_root = data_root.resolve()
    out_dir = resolve_output_dir(args.output_dir, data_root)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        tag = f"_{args.tag}" if args.tag else ""
        ckpt_path = out_dir / f"best_{args.model}{tag}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first or pass --checkpoint."
        )
    print(f"OUTPUT_DIR (resolved): {out_dir}")
    print(f"Loading checkpoint:    {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt["num_classes"]

    _, val_loader, test_loader, mlb = build_dataloaders(
        metadata_csv=str(data_root / "train.csv"),
        audio_dir=str(data_root / "train_audio"),
        soundscapes_dir=str(data_root / "train_soundscapes"),
        soundscapes_csv=str(data_root / "train_soundscapes_labels.csv"),
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
        duration_cache_path=(os.environ.get("BIRDCLEF_DURATION_CACHE") or None),
    )

    if len(mlb.classes_) != num_classes:
        print(
            f"WARNING: class count mismatch (ckpt={num_classes}, "
            f"current={len(mlb.classes_)}). Label space likely changed."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = ckpt.get("model_kwargs", {}) or {}
    model = load_model(ckpt["model_name"], num_classes, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.split == "val":
        # ── Val evaluation: threshold tuning on the same set (exploratory) ───
        y_true, y_score = predict_val_probs(model, val_loader, device)
        auc, per_class = macro_roc_auc(y_true, y_score)
        f1, tuned_thr = macro_f1_tuned(y_true, y_score)
        threshold_source = "tuned_on_val"

    else:  # test
        # ── Test evaluation: use val-tuned thresholds from the checkpoint ────
        # Thresholds are never refit on the test set to prevent leakage.
        if "thresholds" not in ckpt:
            raise ValueError(
                "Checkpoint does not contain 'thresholds'. "
                "Was it saved by the current version of train.py?"
            )
        tuned_thr = np.array(ckpt["thresholds"])
        threshold_source = f"val_epoch_{ckpt.get('epoch', '?')}"
        y_true, y_score = predict_val_probs(model, test_loader, device)
        auc, per_class = macro_roc_auc(y_true, y_score)
        # Apply the val-tuned thresholds to compute test F1.
        f1, _ = macro_f1_tuned(y_true, y_score, thresholds=tuned_thr)
        print(f"Using val-tuned thresholds (epoch {ckpt.get('epoch', '?')}) on test set.")

    valid_mask = ~np.isnan(per_class)
    print(f"\n── Results ({args.split}) ───────────────────────────────────────────")
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Model           : {ckpt['model_name']}  (K={num_classes})")
    print(f"Train-time epoch: {ckpt.get('epoch', '?')}  "
          f"val_auc={ckpt.get('val_auc', float('nan')):.4f}")
    print(f"Macro ROC-AUC   : {auc:.4f}   "
          f"({valid_mask.sum()}/{num_classes} classes with ≥1 pos)")
    print(f"Macro F1 ({threshold_source}): {f1:.4f}")
    if valid_mask.any():
        pc = per_class[valid_mask]
        print(f"Per-class AUC   : mean={pc.mean():.4f} "
              f"min={pc.min():.4f}  max={pc.max():.4f}")

    # ── Persist results ───────────────────────────────────────────────────────
    result: dict = {
        "split": args.split,
        "model": ckpt["model_name"],
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "num_classes": num_classes,
        "macro_auc": auc,
        "macro_f1": f1,
        "threshold_source": threshold_source,
        "class": list(mlb.classes_),
        "per_class_auc": per_class.tolist(),
        "thresholds": tuned_thr.tolist(),
    }

    if args.split == "test":
        tag = f"_{args.tag}" if args.tag else ""
        results_path = out_dir / f"test_results_{args.model}{tag}.json"
        results_path.write_text(json.dumps(result, indent=2))
        print(f"\nTest results written to {results_path}")
    elif args.save_per_class:
        Path(args.save_per_class).write_text(json.dumps(result, indent=2))
        print(f"Wrote per-class AUC to {args.save_per_class}")


if __name__ == "__main__":
    main()
