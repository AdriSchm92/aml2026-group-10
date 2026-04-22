"""Copy the BirdCLEF / Kaggle competition tree into a repo-local stash for fast I/O.

**From an existing path (e.g. Switch / network drive):**
  python scripts/stash_birdclef_data.py --source /path/to/birdclef-2026

**From Kaggle (recommended: API token, needs `kaggle>=1.8.0`, see
https://pypi.org/project/kaggle/):** generate a token at https://www.kaggle.com/settings
(“Generate New Token”), then either ``export KAGGLE_API_TOKEN=...`` or put the same token
in ``~/.kaggle/access_token``. Legacy ``~/.kaggle/kaggle.json`` still works if needed.
  python scripts/stash_birdclef_data.py --kaggle-download

Default destination: <repo>/birdclef_stash/ or BIRDCLEF_STASH_DIR. Local copy uses
``rsync -a`` when available. ``--with-delete`` and ``--no-rsync`` are **ignored** in
``--kaggle-download`` mode.

Then: ``python train.py`` (unset ``DATA_ROOT``) to use the stash. See train.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STASH = REPO_ROOT / "birdclef_stash"
COMPETITION = "birdclef-2026"


def _auth_help() -> str:
    return (
        "Kaggle auth failed. Recommended (kaggle>=1.8.0): API token from "
        "https://www.kaggle.com/settings → KAGGLE_API_TOKEN env or ~/.kaggle/access_token "
        "(https://pypi.org/project/kaggle/). Legacy: ~/.kaggle/kaggle.json"
    )


def _resolve_dest(args: argparse.Namespace) -> Path:
    p = Path(
        args.dest
        or (os.environ.get("BIRDCLEF_STASH_DIR") or "").strip()
        or DEFAULT_STASH
    ).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _promote_if_single_nested_root(dest: Path) -> None:
    if (dest / "train.csv").is_file():
        return
    for sub in list(dest.iterdir()):
        if not sub.is_dir() or not (sub / "train.csv").is_file():
            continue
        for item in sub.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        try:
            sub.rmdir()
        except OSError:
            shutil.rmtree(sub, ignore_errors=True)
        return


def _extract_all_zips_under(dest: Path, max_rounds: int = 48) -> None:
    for _ in range(max_rounds):
        zips = [p for p in dest.rglob("*.zip") if p.is_file()]
        if not zips:
            return
        for zpath in zips:
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(zpath.parent)
            except zipfile.BadZipFile:
                print(f"warning: invalid zip, renaming: {zpath}", file=sys.stderr)
                bad = zpath.with_suffix(".zip.bad")
                if bad.exists():
                    bad.unlink()
                zpath.rename(bad)
                continue
            try:
                zpath.unlink()
            except OSError:
                pass


def _kaggle_with_python_api(dest: Path) -> bool:
    """Use pre-authenticated ``kaggle.api`` (Kaggle 1.8+ API tokens) when available.

    See https://pypi.org/project/kaggle/ — ``import kaggle`` performs token auth;
    a second ``KaggleApi().authenticate()`` can break ``KAGGLE_API_TOKEN`` flows.
    """
    try:
        import kaggle
    except ImportError as e:
        print(
            "error: pip install 'kaggle>=1.8.0' (API tokens need kaggle>=1.8)",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e
    try:
        api = getattr(kaggle, "api", None)
        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
    except Exception as e:  # noqa: BLE001
        print(_auth_help(), file=sys.stderr)
        print(f"{e!r}", file=sys.stderr)
        raise SystemExit(1) from e
    try:
        api.competition_download_files(
            COMPETITION, path=str(dest), force=True, quiet=False
        )
    except Exception as e:  # noqa: BLE001
        print(f"Kaggle API download error: {e!r}, trying kaggle CLI …", file=sys.stderr)
        return False
    return True


def _kaggle_with_cli(dest: Path) -> None:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        print(
            "error: neither Python API download nor `kaggle` on PATH works.",
            file=sys.stderr,
        )
        print(_auth_help(), file=sys.stderr)
        raise SystemExit(1)
    cmd = [kaggle, "competitions", "download", "-c", COMPETITION, "-p", str(dest), "--force"]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, file=sys.stderr, end="")
        print(r.stderr, file=sys.stderr, end="")
        print(_auth_help(), file=sys.stderr)
        raise SystemExit(r.returncode)


def _download_kaggle_to(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if not _kaggle_with_python_api(dest):
        print("Falling back to: kaggle competitions download …", file=sys.stderr)
        _kaggle_with_cli(dest)
    for _ in range(8):
        _extract_all_zips_under(dest)
        _promote_if_single_nested_root(dest)
        if (dest / "train.csv").is_file():
            return
    if not (dest / "train.csv").is_file():
        found = list(dest.rglob("train.csv"))
        print(
            "error: train.csv not at stash root after download. "
            f"Found {len(found)} train.csv path(s) (sample: {found[:2]}).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy or download competition data into birdclef_stash/"
    )
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument(
        "--source",
        default=None,
        help="Existing competition root (contains train.csv) — local copy from disk.",
    )
    m.add_argument(
        "--kaggle-download",
        action="store_true",
        help="Download the full %s comp (Kaggle API token, kaggle>=1.8; see module doc)."
        % COMPETITION,
    )
    p.add_argument(
        "--dest",
        default=None,
        help="Stash path (default: BIRDCLEF_STASH_DIR or <repo>/birdclef_stash/)",
    )
    p.add_argument(
        "--with-delete",
        action="store_true",
        help="(Local --source only) Pass rsync --delete; ignored with --kaggle-download.",
    )
    p.add_argument(
        "--no-rsync",
        action="store_true",
        help="(Local --source only) Use shutil; ignored with --kaggle-download.",
    )
    return p.parse_args()


def _copy_from_source(args: argparse.Namespace, source: Path, dest: Path) -> None:
    if not (source / "train.csv").is_file():
        print(f"error: {source}/train.csv not found. Is this the competition data root?")
        raise SystemExit(1)
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
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest, dirs_exist_ok=True)
    if not (dest / "train.csv").is_file():
        print("error: copy finished but train.csv missing at dest")
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    dest = _resolve_dest(args)
    if args.kaggle_download:
        if args.with_delete or args.no_rsync:
            print(
                "note: --with-delete and --no-rsync are ignored in --kaggle-download mode.",
                file=sys.stderr,
            )
        print(f"dest:   {dest}")
        print("Downloading from Kaggle (this may take a long time, ~16+ GB) …")
        _download_kaggle_to(dest)
    else:
        if not args.source:
            print("error: use --source PATH or --kaggle-download", file=sys.stderr)
            raise SystemExit(2)
        _copy_from_source(args, Path(args.source).resolve(), dest)
    if not (dest / "train.csv").is_file():
        print("error: train.csv missing at dest", file=sys.stderr)
        raise SystemExit(1)
    print("Done. Run: python train.py  (unset DATA_ROOT) to use this stash, or set "
          f"BIRDCLEF_STASH_DIR={dest}.")


if __name__ == "__main__":
    main()
