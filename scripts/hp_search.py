"""Random HP search runner for BirdCLEF 2026.

Loads a YAML config defining the HP candidate grid, samples ``--n_trials``
random configurations, calls ``train.run_training`` in-process for each trial,
and reports the best configuration by val_auc.

Budget control (PROBLEMSETTING.md §HP Tuning — "small candidate grid via random search"):
  - ``--data_subset_min_recordings 200`` restricts to K=69 species (~26% of data).
  - ``--epochs`` is set per-trial from the config (default 3 in the YAML).
  - ``--max_hours`` wall-clock budget; a trial that would exceed it is skipped.
  - Typical budget: 6 trials × 3 epochs × K=69 ≈ 6–8h on a single GPU.

After picking the best HP config, run a final full-data retrain:

    python train.py --model cnn_transformer \\
        --model_kwargs '{"d_model": 256, "n_layers": 4, "n_heads": 8}' \\
        --epochs 15 --warmup_epochs 5 --label_smoothing 0.1

Results are written to ``--output_dir/hp_results_<model>.jsonl`` (one line per trial)
and a summary is printed at the end.

Usage:
    python scripts/hp_search.py --model cnn_transformer --n_trials 6
    python scripts/hp_search.py --model cnn_transformer --config configs/hp_cnn_transformer.yaml \\
        --n_trials 8 --max_hours 10 --output_dir /path/to/runs
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required for HP search. Install with: pip install pyyaml"
    ) from e

from train import parse_args as _parse_train_args, run_training  # noqa: E402


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_config(hp_space: dict, rng: random.Random) -> dict:
    """Uniformly sample one value per HP from the candidate lists."""
    sampled: dict = {}
    for key, candidates in hp_space.items():
        if not isinstance(candidates, list):
            sampled[key] = candidates
        else:
            sampled[key] = rng.choice(candidates)
    return sampled


def _build_train_args(
    base_args: argparse.Namespace,
    trial_config: dict,
    trial_idx: int,
    hp_space: dict,
    model_hp_keys: set[str],
    fixed: dict,
) -> argparse.Namespace:
    """Merge base training args with one trial's sampled HP config."""
    args = copy.deepcopy(base_args)

    # Fixed trial settings from config (epochs, warmup, label_smoothing, etc.)
    for k, v in fixed.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Separate model architecture kwargs from trainer kwargs.
    model_kwargs: dict = {}
    for k, v in trial_config.items():
        if k in model_hp_keys:
            model_kwargs[k] = v
        elif hasattr(args, k):
            setattr(args, k, v)

    args.model_kwargs = json.dumps(model_kwargs) if model_kwargs else None
    args.tag = f"hp{trial_idx:02d}"
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random HP search for BirdCLEF 2026")
    p.add_argument("--model", default="cnn_transformer",
                   help="Model name (must match a file in models/ and a config in configs/).")
    p.add_argument("--config", default=None,
                   help="Path to HP config YAML. Defaults to configs/hp_<model>.yaml.")
    p.add_argument("--n_trials", type=int, default=6,
                   help="Number of random HP configurations to evaluate.")
    p.add_argument("--max_hours", type=float, default=8.0,
                   help="Wall-clock budget in hours. Trials that would start after "
                        "this budget is exhausted are skipped.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for HP sampling (separate from training seed).")
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None,
                   help="Where checkpoints and HP results are written.")
    p.add_argument(
        "--verbose_data",
        action="store_true",
        help="Pass --verbose_data to each training run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config) if args.config else (
        REPO_ROOT / "configs" / f"hp_{args.model}.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"HP config not found: {config_path}. "
            f"Create configs/hp_{args.model}.yaml or pass --config."
        )
    config = _load_config(config_path)

    hp_space: dict = config.get("hp_space", {})
    model_hp_keys: set[str] = set(config.get("model_hp_keys", []))
    fixed: dict = config.get("fixed", {})

    # Determine output dir from a minimal train args namespace.
    base_train_args = _parse_train_args.__wrapped__() if hasattr(_parse_train_args, "__wrapped__") else None
    # Build a minimal Namespace for path resolution — hp_search sets these.
    import train as train_module
    from train import resolve_data_root, resolve_output_dir

    data_root = resolve_data_root(args.data_root)
    data_root = data_root.resolve()
    output_dir = resolve_output_dir(args.output_dir, data_root)

    results_path = output_dir / f"hp_results_{args.model}.jsonl"
    results_path.write_text("")  # truncate / create

    rng = random.Random(args.seed)

    print(f"HP search — model={args.model}, n_trials={args.n_trials}, "
          f"max_hours={args.max_hours:.1f}h, config={config_path}")
    print(f"HP space: {hp_space}")
    print(f"Fixed per-trial settings: {fixed}")
    print(f"Results -> {results_path}\n")

    # Build a default train Namespace so we can override fields.
    sys.argv = ["train.py", "--model", args.model]
    if args.data_root:
        sys.argv += ["--data_root", args.data_root]
    if args.output_dir:
        sys.argv += ["--output_dir", args.output_dir]
    if args.verbose_data:
        sys.argv += ["--verbose_data"]
    base_train_ns = train_module.parse_args()

    wall_start = time.time()
    all_results: list[dict] = []

    for trial_idx in range(args.n_trials):
        elapsed_h = (time.time() - wall_start) / 3600.0
        if elapsed_h >= args.max_hours:
            print(f"\nBudget exhausted ({elapsed_h:.2f}h >= {args.max_hours}h). "
                  f"Stopping after {trial_idx} trials.")
            break

        trial_config = _sample_config(hp_space, rng)
        print(f"\n── Trial {trial_idx + 1}/{args.n_trials} ──────────────────────────────────")
        print(f"   Config: {trial_config}")

        trial_args = _build_train_args(
            base_train_ns, trial_config, trial_idx, hp_space, model_hp_keys, fixed
        )
        # Always use reduced subset for HP search — faster and sufficient for
        # structural HP comparison (PROBLEMSETTING §Scope).
        if not hasattr(trial_args, "data_subset_min_recordings") or \
                trial_args.data_subset_min_recordings is None:
            min_rec = config.get("data_subset_min_recordings", 200)
            trial_args.data_subset_min_recordings = min_rec

        t_trial = time.time()
        try:
            summary = run_training(trial_args)
            best_auc = summary["best_auc"]
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"   Trial {trial_idx + 1} FAILED: {exc}")
            traceback.print_exc()
            best_auc = float("nan")
            status = f"error: {exc}"

        trial_elapsed = time.time() - t_trial
        row = {
            "trial": trial_idx,
            "config": trial_config,
            "best_auc": best_auc,
            "status": status,
            "elapsed_s": trial_elapsed,
        }
        all_results.append(row)
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"   -> best_auc={best_auc:.4f}  elapsed={trial_elapsed/60:.1f}min")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = [r for r in all_results if not math.isnan(r["best_auc"])]
    if not valid:
        print("\nNo successful trials. Check logs above.")
        return

    best = max(valid, key=lambda r: r["best_auc"])
    total_h = (time.time() - wall_start) / 3600.0
    print(f"\n{'='*60}")
    print(f"HP search complete — {len(valid)}/{len(all_results)} trials succeeded "
          f"in {total_h:.2f}h")
    print(f"Best trial #{best['trial']}: val_auc={best['best_auc']:.4f}")
    print(f"Best config: {best['config']}")
    print(f"\nFinal retrain command:")

    # Separate model kwargs from trainer kwargs for the final run command.
    model_kwargs = {k: v for k, v in best["config"].items() if k in model_hp_keys}
    trainer_kwargs = {k: v for k, v in best["config"].items() if k not in model_hp_keys}
    final_cmd = [f"python train.py --model {args.model}"]
    if model_kwargs:
        final_cmd.append(f"    --model_kwargs '{json.dumps(model_kwargs)}'")
    for k, v in trainer_kwargs.items():
        final_cmd.append(f"    --{k} {v}")
    final_cmd.append("    --epochs 15  # full-data retrain")
    print("\n".join(final_cmd))
    print(f"\nResults log: {results_path}")


if __name__ == "__main__":
    main()
