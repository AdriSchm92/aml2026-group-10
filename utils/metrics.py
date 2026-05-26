"""Multi-label metrics for BirdCLEF.

macro_roc_auc      : macro-averaged ROC-AUC over classes with ≥1 pos and ≥1 neg.
macro_f1_tuned     : macro-averaged F1 with per-class threshold sweep on the
                     same data (use on val only; apply the thresholds to test).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_roc_auc(y_true: np.ndarray, y_score: np.ndarray):
    """Return (macro_auc, per_class_auc array).

    Degenerate classes (all-0 or all-1 in y_true) are skipped from the macro
    average and reported as NaN in per_class_auc. This is the robust version
    of sklearn's ``average='macro'`` which otherwise raises.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    num_classes = y_true.shape[1]
    per_class = np.full(num_classes, np.nan)
    valid_aucs = []
    for k in range(num_classes):
        yt = y_true[:, k]
        pos = int(yt.sum())
        if pos == 0 or pos == len(yt):
            continue
        auc = roc_auc_score(yt, y_score[:, k])
        per_class[k] = auc
        valid_aucs.append(auc)
    macro = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro, per_class


def macro_f1_at_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """Apply fixed per-class thresholds directly; return macro F1. No search.

    Use this for test evaluation to apply val-tuned thresholds without any
    test-set search (which would be leakage).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    num_classes = y_true.shape[1]
    f1s = [
        f1_score(
            y_true[:, k],
            (y_score[:, k] >= thresholds[k]).astype(np.int32),
            zero_division=0,
        )
        for k in range(num_classes)
    ]
    return float(np.mean(f1s)) if f1s else float("nan")


def macro_f1_tuned(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
):
    """Per-class threshold sweep → macro F1. Returns (macro_f1, thresholds).

    ``thresholds[k]`` is the argmax-F1 threshold for class k. Classes with zero
    positives keep the default 0.5 and contribute 0 to the macro average (not
    skipped, to stay consistent with the multi-label macro convention).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    num_classes = y_true.shape[1]
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_thr = np.full(num_classes, 0.5)
    f1s = []
    for k in range(num_classes):
        yt = y_true[:, k]
        if yt.sum() == 0:
            f1s.append(0.0)
            continue
        best = 0.0
        best_t = 0.5
        for t in thresholds:
            yp = (y_score[:, k] >= t).astype(np.int32)
            f = f1_score(yt, yp, zero_division=0)
            if f > best:
                best = f
                best_t = t
        best_thr[k] = best_t
        f1s.append(best)
    macro = float(np.mean(f1s)) if f1s else float("nan")
    return macro, best_thr
