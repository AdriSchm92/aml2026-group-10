import hashlib
import json
import os
import tempfile
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torchaudio.transforms as T


# ─────────────────────────────────────────────
# PREPROCESSING CONSTANTS
# Confirmed from EDA: 32000 Hz, 5s → (128, 313), 10s → (128,626)
# ─────────────────────────────────────────────
SAMPLE_RATE   = 32000
CLIP_DURATION = 5 #10
N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 1024
F_MIN         = 50       # ignore very low frequencies (wind noise)
F_MAX         = 14000    # upper bound relevant for bird calls

# Per-file librosa.get_duration is slow on network mounts; cache in repo .cache/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ─────────────────────────────────────────────
# SPECTROGRAM CACHE HELPERS
# Pre-computed mel spectrograms saved as float16 .npy files on local disk.
# Eliminates librosa.load + mel computation (50–200ms/sample) on every epoch.
# Set spec_cache_dir to a LOCAL path (e.g. /tmp/birdclef_specs) — NOT a network
# mount, or the cache defeats its own purpose.
# ─────────────────────────────────────────────

def _spec_cache_key(file_path: str, offset: float) -> str:
    """Stable MD5 key from absolute path + offset seconds."""
    raw = f"{os.path.abspath(file_path)}@{offset:.3f}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_spec_cache(cache_dir: str, key: str) -> np.ndarray | None:
    path = os.path.join(cache_dir, key + ".npy")
    try:
        return np.load(path).astype(np.float32)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _save_spec_cache(cache_dir: str, key: str, arr: np.ndarray) -> None:
    """Atomic write so concurrent DataLoader workers never corrupt a cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, key + ".npy")
    if os.path.exists(dest):
        return
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix="spec.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr.astype(np.float16))
        os.replace(tmp, dest)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _default_duration_cache_path() -> str:
    return os.path.join(REPO_ROOT, ".cache", "birdclef", "train_audio_durations.json")


def _stat_sig(path: str) -> tuple[int, int]:
    st = os.stat(path)
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    return int(mtime_ns), int(st.st_size)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_json(path: str, data: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=d or None, prefix="durations.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=0, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class BirdCLEFDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF 2026.

    Handles two data sources:
      1. train_audio/   — clean single-species recordings (single-label)
      2. train_soundscapes/ — real-world multi-species soundscapes (multi-label)

    Each item is a (spectrogram, label_vector) pair where:
      - spectrogram : torch.Tensor of shape (1, N_MELS, T')
      - label_vector: torch.Tensor of shape (K,) — multi-hot binary vector
    """

    def __init__(self, samples, label_encoder, augment=False, spec_cache_dir=None):
        """
        Args:
            samples        : list of dicts, each with keys:
                               'file_path' : str   — path to .ogg file
                               'offset'    : float — start time in seconds
                               'labels'    : list  — list of species label strings
            label_encoder  : fitted MultiLabelBinarizer mapping species → index
            augment        : if True, apply SpecAugment (training set only)
            spec_cache_dir : local directory for float16 .npy spectrogram cache.
                             Set to a fast LOCAL path (e.g. /tmp/birdclef_specs).
                             Eliminates librosa.load + mel recomputation every epoch.
        """
        self.samples        = samples
        self.label_encoder  = label_encoder
        self.augment        = augment
        self.n_classes      = len(label_encoder.classes_)
        self._spec_cache_dir = spec_cache_dir

        # O(1) label → index lookup (np.where scan is O(K) per label per item)
        self._class_set    = set(label_encoder.classes_)
        self._class_to_idx = {c: i for i, c in enumerate(label_encoder.classes_)}

        # SpecAugment: one frequency mask + one time mask per sample.
        self.freq_mask = T.FrequencyMasking(freq_mask_param=24)
        self.time_mask = T.TimeMasking(time_mask_param=64)

    def __len__(self):
        return len(self.samples)

    def _compute_mel_db(self, sample) -> np.ndarray:
        """Load audio chunk and return normalised (128, 313) float32 mel spectrogram."""
        y, _ = librosa.load(
            sample['file_path'],
            sr       = SAMPLE_RATE,
            offset   = sample['offset'],
            duration = CLIP_DURATION,
            mono     = True,
        )
        target_len = SAMPLE_RATE * CLIP_DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        mel    = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        return mel_db.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ── 1–4. Load / compute normalised mel spectrogram ───────────────────
        # If a spec_cache_dir is set, serve from float16 .npy cache on first hit
        # and populate it on miss.  Converts librosa's 50–200ms per sample into
        # a ~1ms numpy load after the first epoch.
        if self._spec_cache_dir is not None:
            key    = _spec_cache_key(sample['file_path'], sample['offset'])
            mel_db = _load_spec_cache(self._spec_cache_dir, key)
            if mel_db is None:
                mel_db = self._compute_mel_db(sample)
                _save_spec_cache(self._spec_cache_dir, key, mel_db)
        else:
            mel_db = self._compute_mel_db(sample)

        # ── 5. Convert to tensor with channel dim → (1, 128, 313) ────────────
        spec = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

        # ── 6. Apply SpecAugment (training only) ─────────────────────────────
        if self.augment:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        # ── 7. Build multi-hot label vector ──────────────────────────────────
        # Works for both single-label (train_audio) and multi-label (soundscapes)
        # since sample['labels'] is always a list, even if it has one element
        known_labels = [l for l in sample['labels'] if l in self._class_set]
        label_vec    = torch.zeros(self.n_classes, dtype=torch.float32)
        if known_labels:
            indices = [self._class_to_idx[l] for l in known_labels]
            label_vec[indices] = 1.0

        return spec, label_vec


# ─────────────────────────────────────────────
# SAMPLE LIST BUILDERS
# One function per data source, both return
# a list of dicts with the same structure so
# they can be concatenated cleanly.
# ─────────────────────────────────────────────

def build_samples_from_train_audio(
    df,
    audio_dir,
    duration_cache_path: str | None = None,
    verbose_data: bool = False,
    split_label: str = "",
):
    """
    Builds sample list from train_audio/ (clean single-species recordings).

    Each recording is chunked into non-overlapping 5s windows.
    Label is a single-element list to stay consistent with the soundscape format.

    Args:
        df        : pd.DataFrame with columns ['filename', 'primary_label']
        audio_dir : path to train_audio/ folder
        duration_cache_path : JSON cache of path -> duration + mtime/size (avoids
            repeated librosa.get_duration on slow / network filesystems).
        verbose_data : print cache miss count

    Returns:
        list of dicts with keys: file_path, offset, labels
    """
    cache_path = duration_cache_path or _default_duration_cache_path()
    cache = _load_json(cache_path)
    prefix = f"[{split_label}] " if split_label else ""
    if cache:
        print(f"{prefix}duration cache: {len(cache)} existing entries in {cache_path}")
    n_hit, n_miss = 0, 0
    samples = []
    for _, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        if not os.path.exists(file_path):
            continue
        abspath = os.path.normpath(os.path.abspath(file_path))
        mtime_ns, size = _stat_sig(abspath)
        ent = cache.get(abspath)
        if (
            ent
            and int(ent.get("mtime_ns", -1)) == mtime_ns
            and int(ent.get("size", -1)) == size
            and "duration" in ent
        ):
            duration = float(ent["duration"])
            n_hit += 1
        else:
            duration = float(librosa.get_duration(path=file_path))
            cache[abspath] = {
                "duration": duration,
                "mtime_ns": mtime_ns,
                "size": size,
            }
            n_miss += 1
        n_chunks = max(1, int(duration // CLIP_DURATION))
        for chunk_idx in range(n_chunks):
            samples.append({
                'file_path' : file_path,
                'offset'    : chunk_idx * CLIP_DURATION,
                'labels'    : [row['primary_label']],  # single-label, wrapped in list
            })
    _atomic_write_json(cache_path, cache)
    print(
        f"{prefix}duration cache: {len(cache)} total entries  "
        f"(hits={n_hit}, misses={n_miss})  -> {cache_path}"
    )
    if verbose_data and n_miss:
        print(f"{prefix}duration cache: {n_miss} files re-scanned this run")
    return samples


def build_samples_from_soundscapes(labels_csv, soundscapes_dir):
    """
    Builds sample list from labeled train_soundscapes/.

    The labels CSV has columns: filename, start, end, primary_label
    where primary_label is a semicolon-separated list of species codes
    for that 5-second segment (already pre-segmented by expert annotators).

    Args:
        labels_csv      : path to train_soundscapes_labels.csv
        soundscapes_dir : path to train_soundscapes/ folder

    Returns:
        list of dicts with keys: file_path, offset, labels
    """
    df      = pd.read_csv(labels_csv)
    samples = []
    for _, row in df.iterrows():
        file_path = os.path.join(soundscapes_dir, row['filename'])
        if not os.path.exists(file_path):
            continue
        # convert HH:MM:SS start time to seconds
        h, m, s  = str(row['start']).split(':')
        offset_s = int(h) * 3600 + int(m) * 60 + float(s)
        # parse semicolon-separated multi-label species list
        species_list = [s.strip() for s in str(row['primary_label']).split(';')]
        samples.append({
            'file_path' : file_path,
            'offset'    : offset_s,
            'labels'    : species_list,
        })
    return samples


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def build_dataloaders(
    metadata_csv    : str,
    audio_dir       : str,
    soundscapes_dir : str,
    soundscapes_csv : str,
    val_size        : float = 0.15,
    test_size       : float = 0.15,
    batch_size      : int   = 32,
    num_workers     : int   = 4,
    random_state    : int   = 42,
    duration_cache_path: str | None = None,
    verbose_data    : bool  = False,
    min_recordings  : int | None = None,
    spec_cache_dir  : str | None = None,
    soundscape_val_files: set[str] | frozenset[str] | None = None,
    taxonomy_csv    : str | None = None,
):
    """Builds train / val / test DataLoaders from both data sources.

    Split strategy (PROBLEMSETTING.md §Evaluation Protocol):
      - Deterministic 70 / 15 / 15 stratified split by primary_label.
      - Soundscape segments are added to the training set only.
      - Val and test sets come exclusively from clean single-label train_audio.
      - mlb is fit on the *full* df so all species are covered.

    Args:
        metadata_csv    : path to train.csv
        audio_dir       : path to train_audio/ folder
        soundscapes_dir : path to train_soundscapes/ folder
        soundscapes_csv : path to train_soundscapes_labels.csv
        val_size        : fraction of total recordings held out for validation
        test_size       : fraction of total recordings held out for final testing
        batch_size      : DataLoader batch size
        num_workers     : parallel workers for data loading
        random_state    : random seed — same seed guarantees identical splits
        duration_cache_path: JSON file for train_audio duration cache
        verbose_data    : extra cache / I/O output
        min_recordings  : if set, restrict to species with at least this many
                          recordings (PROBLEMSETTING §Scope fallback, K=69 at
                          threshold 200). Applied before splitting; mlb is
                          fit on the filtered df so K reflects the subset.
        spec_cache_dir  : local directory for pre-computed spectrogram cache.
                          Must be a fast LOCAL path (e.g. /tmp/birdclef_specs),
                          not a network mount. See BIRDCLEF_SPEC_CACHE env var.
        soundscape_val_files: file paths held out for soundscape val — excluded
                          from training soundscape segments (no leakage).
        taxonomy_csv    : if set, fit MLB on all species in this file (K=234
                          for BirdCLEF 2026) instead of train.csv primary labels
                          only (K=206). Used for Kaggle-track training.

    Returns:
        train_loader, val_loader, test_loader, label_encoder (MultiLabelBinarizer)
    """

    # ── 1. Load train_audio metadata ─────────────────────────────────────────
    df = pd.read_csv(metadata_csv)
    print(f"train_audio — recordings: {len(df)}, species: {df['primary_label'].nunique()}")

    # ── 2. Optional species-count filter (HP search / compute budget) ─────────
    if min_recordings is not None and min_recordings > 1:
        counts_all = df["primary_label"].value_counts()
        keep = counts_all[counts_all >= min_recordings].index
        df = df[df["primary_label"].isin(keep)].reset_index(drop=True)
        print(
            f"min_recordings={min_recordings}: retained {len(keep)} species, "
            f"{len(df)} recordings"
        )

    # ── 3. Stratified 3-way split ─────────────────────────────────────────────
    # Classes with only 1 recording cannot be stratified — move to train only.
    counts = df["primary_label"].value_counts()
    singleton_species = counts[counts < 2].index
    df_main = df[~df["primary_label"].isin(singleton_species)]
    df_singletons = df[df["primary_label"].isin(singleton_species)]

    # First cut: carve out test set.
    train_val_df, test_df = train_test_split(
        df_main,
        test_size=test_size,
        stratify=df_main["primary_label"],
        random_state=random_state,
    )

    # Second cut: split remainder into train / val.
    # Adjust val fraction so the absolute val size equals val_size of full df.
    adjusted_val = val_size / (1.0 - test_size)
    # Guard against rounding — clamp to a valid range.
    adjusted_val = max(0.01, min(adjusted_val, 0.99))
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val,
        stratify=train_val_df["primary_label"],
        random_state=random_state,
    )

    if len(df_singletons) > 0:
        train_df = pd.concat([train_df, df_singletons], ignore_index=True)
        print(
            f"Singleton classes moved to train only: "
            f"{len(df_singletons)} recordings, {df_singletons['primary_label'].nunique()} species"
        )

    total = len(train_df) + len(val_df) + len(test_df)
    print(
        f"Split — train: {len(train_df)} ({len(train_df)/total:.0%}), "
        f"val: {len(val_df)} ({len(val_df)/total:.0%}), "
        f"test: {len(test_df)} ({len(test_df)/total:.0%})  "
        f"[seed={random_state}]"
    )

    # ── 4. Fit label encoder ──────────────────────────────────────────────────
    # Default: train.csv primary labels (K=206, PROBLEMSETTING report track).
    # taxonomy_csv: full competition taxonomy (K=234, Kaggle track).
    mlb = MultiLabelBinarizer()
    if taxonomy_csv:
        tax_df = pd.read_csv(taxonomy_csv)
        label_col = None
        for col in ("primary_label", "ebird_code", "species_code"):
            if col in tax_df.columns:
                label_col = col
                break
        if label_col is None:
            raise ValueError(
                f"taxonomy_csv {taxonomy_csv} needs primary_label (or ebird_code) column; "
                f"got {list(tax_df.columns)}"
            )
        mlb.fit([[s] for s in sorted(tax_df[label_col].unique())])
        print(f"MLB fit on taxonomy ({taxonomy_csv}, col={label_col}): K={len(mlb.classes_)}")
    else:
        mlb.fit([[s] for s in sorted(df['primary_label'].unique())])
        print(f"Number of classes K: {len(mlb.classes_)}")

    # ── 5. Build sample lists ─────────────────────────────────────────────────
    train_audio_samples = build_samples_from_train_audio(
        train_df, audio_dir, duration_cache_path, verbose_data, "train"
    )
    exclude = soundscape_val_files or set()
    soundscape_samples = [
        s for s in build_samples_from_soundscapes(soundscapes_csv, soundscapes_dir)
        if s["file_path"] not in exclude
    ]
    if exclude:
        print(
            f"Soundscape training segments: {len(soundscape_samples)} "
            f"(excluded {len(exclude)} val file(s))"
        )
    val_samples = build_samples_from_train_audio(
        val_df, audio_dir, duration_cache_path, verbose_data, "val"
    )
    test_samples = build_samples_from_train_audio(
        test_df, audio_dir, duration_cache_path, verbose_data, "test"
    )

    # Training set = clean recordings + real-world soundscapes
    train_samples = train_audio_samples + soundscape_samples
    print(
        f"Training samples  — audio chunks: {len(train_audio_samples)}, "
        f"soundscape segments: {len(soundscape_samples)}, "
        f"total: {len(train_samples)}"
    )
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples      : {len(test_samples)}")

    # ── 6. Build Dataset objects ──────────────────────────────────────────────
    if spec_cache_dir:
        print(f"Spectrogram cache : {spec_cache_dir}")
    train_dataset = BirdCLEFDataset(train_samples, mlb, augment=True,  spec_cache_dir=spec_cache_dir)
    val_dataset   = BirdCLEFDataset(val_samples,   mlb, augment=False, spec_cache_dir=spec_cache_dir)
    test_dataset  = BirdCLEFDataset(test_samples,  mlb, augment=False, spec_cache_dir=spec_cache_dir)

    # ── 7. Build DataLoaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = num_workers > 0,
    )

    return train_loader, val_loader, test_loader, mlb