"""RF baseline (PROBLEMSETTING.md §Baselines — "Simple baseline").

MFCC mean+std feature extraction + OneVsRestClassifier(RandomForest).
No deep learning, no spectrogram. Trained independently per class (OvR).
"""
from __future__ import annotations

import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing.data_pipeline import CLIP_DURATION, SAMPLE_RATE  # noqa: E402


def extract_features(file_path: str, offset: float, n_mfcc: int = 40) -> np.ndarray:
    """5-second chunk → (2*n_mfcc,) vector: MFCC mean + std over time axis."""
    y, _ = librosa.load(
        file_path, sr=SAMPLE_RATE, offset=offset, duration=float(CLIP_DURATION), mono=True
    )
    target = SAMPLE_RATE * CLIP_DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def build_model(
    n_estimators: int = 100,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> OneVsRestClassifier:
    return OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    )
