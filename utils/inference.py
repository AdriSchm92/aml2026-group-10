"""Inference helpers: val-loader prediction + file-level 5s chunk max-pool.

Reuses the exact mel parameters from data.preprocessing.data_pipeline so
training and inference see identical spectrograms.
"""
from __future__ import annotations

from typing import Iterable

import librosa
import numpy as np
import torch

from data.preprocessing.data_pipeline import (
    CLIP_DURATION,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)


def waveform_to_spec(y: np.ndarray) -> torch.Tensor:
    """5-second waveform → (1, N_MELS, T') tensor. Pads short input with zeros."""
    target = SAMPLE_RATE * CLIP_DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def predict_val_probs(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
):
    """Run the model over a val DataLoader and return stacked (y_true, y_score).

    Shapes: y_true = (N, K) int/float, y_score = (N, K) float sigmoid probs.
    """
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_score_chunks: list[np.ndarray] = []
    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        logits = model(specs)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_true_chunks.append(labels.detach().cpu().numpy())
        y_score_chunks.append(probs)
    return np.concatenate(y_true_chunks), np.concatenate(y_score_chunks)


@torch.no_grad()
def predict_clip_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
):
    """Chunk-level predictions max-pooled to clip (recording) level.

    Groups 5s chunks by source file and takes elementwise max across chunks,
    matching PROBLEMSETTING.md §Input Representation clip-level aggregation.

    Returns:
        y_true_clip  : (N_files, K) — per-file labels (union over chunks)
        y_score_clip : (N_files, K) — per-file scores (max over chunks)
    """
    from collections import defaultdict

    y_true, y_score = predict_val_probs(model, loader, device)
    samples = loader.dataset.samples

    file_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        file_to_idxs[s["file_path"]].append(i)

    y_true_clip, y_score_clip = [], []
    for fp in sorted(file_to_idxs):
        idxs = file_to_idxs[fp]
        y_true_clip.append(y_true[idxs].max(axis=0))
        y_score_clip.append(y_score[idxs].max(axis=0))
    return np.array(y_true_clip), np.array(y_score_clip)


@torch.no_grad()
def predict_file(
    model: torch.nn.Module,
    file_path: str,
    device: torch.device,
    chunk_batch_size: int = 16,
):
    """Chunk a recording into non-overlapping 5s windows, forward, max-pool.

    Returns (clip_probs, window_probs):
        clip_probs   : (K,) max over windows  — the clip-level prediction
        window_probs : (n_windows, K)         — per-window probs for inspection
    """
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    target = SAMPLE_RATE * CLIP_DURATION
    n_windows = max(1, int(np.ceil(len(y) / target)))

    specs = []
    for i in range(n_windows):
        chunk = y[i * target : (i + 1) * target]
        specs.append(waveform_to_spec(chunk))
    batch = torch.stack(specs).to(device, non_blocking=True)

    window_probs_chunks = []
    for start in range(0, batch.shape[0], chunk_batch_size):
        logits = model(batch[start : start + chunk_batch_size])
        window_probs_chunks.append(torch.sigmoid(logits))
    window_probs = torch.cat(window_probs_chunks, dim=0).detach().cpu().numpy()
    clip_probs = window_probs.max(axis=0)
    return clip_probs, window_probs
