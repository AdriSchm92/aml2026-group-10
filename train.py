"""Training harness for BirdCLEF 2026.

Paths: ``--data_root`` and ``DATA_ROOT`` override everything else. If both are unset, a
valid **local stash** is used: ``birdclef_stash/`` at the repo root (or ``BIRDCLEF_STASH_DIR``) when
``train.csv`` is present. Populate the stash from a slow or remote tree with
``python scripts/stash_birdclef_data.py --source /path/to/birdclef-2026`` or
``--kaggle-download`` with a Kaggle **API token** (``KAGGLE_API_TOKEN`` or
``~/.kaggle/access_token``; ``kaggle>=1.8.0`` — see that script and https://pypi.org/project/kaggle/). Otherwise
resolution falls back to Kaggle input or ``data/raw/``.

``TRAINING_OUTPUT_DIR`` / ``TRAINING_OUTPUT_SUBDIR`` / ``--output_dir``: checkpoints + metrics.
``TRAINING_OUTPUT_SUBDIR`` default ``aml2026-group10-runs`` on Renku under ``kaggle-data/`` parents.

Duration cache (``librosa.get_duration`` on many files) lives in ``.cache/birdclef/``; see
``data_pipeline``.

Kaggle: auto-detects /kaggle/input and /kaggle/working. Renku: often
``--data_root /home/renku/work/kaggle-data/...`` **or** sync into ``birdclef_stash/`` and omit ``DATA_ROOT``.

``python train.py --model cnn_baseline`` — smoke: ``--limit_train_batches 4 --epochs 1``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import build_dataloaders  # noqa: E402
from models.registry import available_models, load_model  # noqa: E402
from utils.inference import predict_val_probs  # noqa: E402
from utils.metrics import macro_f1_tuned, macro_roc_auc  # noqa: E402


class TrainingRunSummary(TypedDict):
    best_auc: float
    checkpoint: str
    metrics: str


def _default_stash_path() -> Path:
    env = (os.environ.get("BIRDCLEF_STASH_DIR") or "").strip()
    return Path(env) if env else (REPO_ROOT / "birdclef_stash")


def _valid_stash_data_root() -> Path | None:
    p = _default_stash_path()
    if (p / "train.csv").is_file():
        return p.resolve()
    return None


def _is_stash_path(resolved: Path) -> bool:
    stash = _valid_stash_data_root()
    if stash is None:
        return False
    try:
        return resolved == stash
    except OSError:
        return False


def resolve_data_root(user_arg: str | None) -> Path:
    """Locate the data root. Explicit ``--data_root`` / ``DATA_ROOT`` beat the local stash."""
    tried: list[str] = []
    if user_arg:
        p = Path(user_arg)
        if not p.exists():
            raise FileNotFoundError(f"--data_root does not exist: {user_arg}")
        r = p.resolve()
        if _is_stash_path(r):
            print(f"Using local stash (from --data_root): {r}")
        return r
    data_root = (os.environ.get("DATA_ROOT") or "").strip()
    if data_root:
        p = Path(data_root)
        if not p.exists():
            raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
        r = p.resolve()
        if _is_stash_path(r):
            print(f"Using local stash (from DATA_ROOT): {r}")
        return r
    stash = _valid_stash_data_root()
    if stash is not None:
        print(f"Using local stash: {stash}")
        return stash
    tried.append(f"stash:{_default_stash_path()}")
    for c in ("/kaggle/input/birdclef-2026", str(REPO_ROOT / "data" / "raw")):
        if c and Path(c).exists():
            return Path(c)
        if c:
            tried.append(c)
    raise FileNotFoundError(
        "No data root found. Pass --data_root, set DATA_ROOT, run "
        "scripts/stash_birdclef_data.py, or link data under data/raw/. "
        f"Tried: {tried}"
    )


def resolve_output_dir(cli_path: str | None, data_root: Path) -> Path:
    """Writables: CLI, then TRAINING_OUTPUT_DIR, then Renku+kaggle-data heuristic, else Kaggle, else ./runs."""
    env_out = (os.environ.get("TRAINING_OUTPUT_DIR") or "").strip()
    if cli_path:
        out = Path(cli_path)
    elif env_out:
        out = Path(env_out)
    elif Path("/home/renku").exists() and "kaggle-data" in str(data_root):
        sub = (os.environ.get("TRAINING_OUTPUT_SUBDIR") or "aml2026-group10-runs").strip()
        out = data_root.parent / sub
    elif Path("/kaggle/working").exists() and os.access("/kaggle/working", os.W_OK):
        out = Path("/kaggle/working") / "runs"
    else:
        out = REPO_ROOT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BirdCLEF 2026 training harness")
    p.add_argument("--data_root", default=None,
                   help="Root with train.csv, train_audio/, train_soundscapes/, "
                        "train_soundscapes_labels.csv. Auto-detected if omitted.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write checkpoints + metrics. Auto-detected.")
    p.add_argument("--model", default="cnn_baseline",
                   help="Name of a file in models/ (e.g. cnn_baseline, "
                        "vit_baseline, cnn_transformer). Each such file must "
                        "define build_model(num_classes) -> nn.Module.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision (ignored on CPU).")
    p.add_argument("--limit_train_batches", type=int, default=None,
                   help="Cap batches/epoch for smoke tests.")
    p.add_argument("--limit_val_batches", type=int, default=None,
                   help="Cap val batches (smoke tests).")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tag", default=None,
                   help="Optional suffix for checkpoint filename.")
    p.add_argument(
        "--verbose_data",
        action="store_true",
        help="Extra duration-cache / dataloader details.",
    )
    return p.parse_args()


def run_epoch(model, loader, optimizer, criterion, scaler, device, args):
    model.train()
    running_loss = 0.0
    n = 0
    use_amp = args.amp and device.type == "cuda"
    for bi, (specs, labels) in enumerate(loader):
        if args.limit_train_batches is not None and bi >= args.limit_train_batches:
            break
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(specs)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        running_loss += loss.item()
        n += 1
    return running_loss / max(1, n)


def cap_loader(loader, limit):
    """Materialise up to ``limit`` batches from a loader (smoke tests)."""
    if limit is None:
        return loader
    batches = []
    for i, b in enumerate(loader):
        if i >= limit:
            break
        batches.append(b)
    return batches


def main() -> TrainingRunSummary:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = resolve_data_root(args.data_root)
    data_root = data_root.resolve()
    output_dir = resolve_output_dir(args.output_dir, data_root)
    print(f"DATA_ROOT  : {data_root}")
    print(f"OUTPUT_DIR : {output_dir}")
    if "/home/renku" in str(data_root) and "kaggle-data" in str(data_root):
        print(
            "Note: kaggle-data on a network mount; long startup is reduced by "
            "the duration cache under .cache/birdclef/ in the repo."
        )
    print(f"ARGS       : {vars(args)}")

    t0 = time.perf_counter()
    train_loader, val_loader, mlb = build_dataloaders(
        metadata_csv=str(data_root / "train.csv"),
        audio_dir=str(data_root / "train_audio"),
        soundscapes_dir=str(data_root / "train_soundscapes"),
        soundscapes_csv=str(data_root / "train_soundscapes_labels.csv"),
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
        duration_cache_path=(os.environ.get("BIRDCLEF_DURATION_CACHE") or None),
        verbose_data=args.verbose_data,
    )
    print(f"build_dataloaders: {time.perf_counter() - t0:.2f}s  (DataLoader num_workers={args.num_workers})")

    num_classes = len(mlb.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"K          : {num_classes}")
    print(f"device     : {device}")
    print(f"models     : available={available_models()}  selected={args.model}")

    model = load_model(args.model, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    tag = f"_{args.tag}" if args.tag else ""
    ckpt_path = output_dir / f"best_{args.model}{tag}.pt"
    metrics_path = output_dir / f"metrics_{args.model}{tag}.jsonl"
    metrics_path.write_text("")
    best_auc = -1.0

    val_batches = cap_loader(val_loader, args.limit_val_batches)

    t_warm = time.perf_counter()
    _ = next(iter(train_loader))
    print(f"first training batch: {time.perf_counter() - t_warm:.2f}s")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args
        )
        scheduler.step()

        y_true, y_score = predict_val_probs(model, val_batches, device)
        auc, per_class_auc = macro_roc_auc(y_true, y_score)
        f1, thresholds = macro_f1_tuned(y_true, y_score)
        dt = time.time() - t0
        print(
            f"[epoch {epoch:03d}] "
            f"loss={train_loss:.4f}  val_auc={auc:.4f}  val_f1={f1:.4f}  "
            f"time={dt:.1f}s"
        )

        with metrics_path.open("a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": auc,
                "val_f1": f1,
                "time_s": dt,
                "lr": optimizer.param_groups[0]["lr"],
            }) + "\n")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "num_classes": num_classes,
                    "classes": list(mlb.classes_),
                    "thresholds": thresholds.tolist(),
                    "epoch": epoch,
                    "val_auc": auc,
                    "val_f1": f1,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path.name} (val_auc={auc:.4f})")

    print(f"Done. Best val AUC: {best_auc:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Metrics   : {metrics_path}")
    return {
        "best_auc": best_auc,
        "checkpoint": str(ckpt_path),
        "metrics": str(metrics_path),
    }


if __name__ == "__main__":
    import traceback

    try:
        from utils.notify import training_event
    except Exception:
        def training_event(_title: str, _body: str = "") -> None:  # noqa: ARG001
            pass

    try:
        out = main()
    except KeyboardInterrupt:
        training_event("interrupted (KeyboardInterrupt)", "Stopped by user.")
        raise SystemExit(130) from None
    except Exception as e:
        tb = traceback.format_exc()
        training_event("training error", f"{type(e).__name__}: {e}\n{tb[-1500:]}")
        raise
    else:
        training_event(
            "finished",
            f"best_auc={out['best_auc']:.4f}\ncheckpoint={out['checkpoint']}\nmetrics={out['metrics']}",
        )
