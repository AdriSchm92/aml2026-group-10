"""Standalone evaluation on the val split for a saved checkpoint.

Loads the model via models.registry (same as train.py) and reuses the val
loader from data.preprocessing.data_pipeline. Prints macro AUC + tuned
macro F1 + per-class stats.

Usage:
  python evaluate.py --checkpoint runs/best_resnet18.pt
  python evaluate.py --checkpoint /kaggle/working/runs/best_resnet18.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import build_dataloaders  # noqa: E402
from models.registry import load_model  # noqa: E402
from train import resolve_data_root  # noqa: E402
from utils.inference import predict_val_probs  # noqa: E402
from utils.metrics import macro_f1_tuned, macro_roc_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_per_class", default=None,
                   help="Optional path to write per-class AUC JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_classes = ckpt["num_classes"]

    _, val_loader, mlb = build_dataloaders(
        metadata_csv=str(data_root / "train.csv"),
        audio_dir=str(data_root / "train_audio"),
        soundscapes_dir=str(data_root / "train_soundscapes"),
        soundscapes_csv=str(data_root / "train_soundscapes_labels.csv"),
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
    )

    if len(mlb.classes_) != num_classes:
        print(
            f"WARNING: class count mismatch (ckpt={num_classes}, "
            f"current={len(mlb.classes_)}). Label space likely changed."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt["model_name"], num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_score = predict_val_probs(model, val_loader, device)
    auc, per_class = macro_roc_auc(y_true, y_score)
    f1, _ = macro_f1_tuned(y_true, y_score)

    valid_mask = ~np.isnan(per_class)
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Model           : {ckpt['model_name']}  (K={num_classes})")
    print(f"Train-time epoch: {ckpt.get('epoch', '?')}  "
          f"val_auc={ckpt.get('val_auc', float('nan')):.4f}")
    print(f"Macro ROC-AUC   : {auc:.4f}   "
          f"({valid_mask.sum()}/{num_classes} classes with ≥1 pos in val)")
    print(f"Macro F1 tuned  : {f1:.4f}")
    if valid_mask.any():
        pc = per_class[valid_mask]
        print(f"Per-class AUC   : mean={pc.mean():.4f} "
              f"min={pc.min():.4f}  max={pc.max():.4f}")

    if args.save_per_class:
        out = {
            "class": list(mlb.classes_),
            "per_class_auc": per_class.tolist(),
            "macro_auc": auc,
            "macro_f1": f1,
        }
        Path(args.save_per_class).write_text(json.dumps(out, indent=2))
        print(f"Wrote per-class AUC to {args.save_per_class}")


if __name__ == "__main__":
    main()
