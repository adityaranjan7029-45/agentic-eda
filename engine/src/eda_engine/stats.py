"""Small, dependency-free statistics helpers shared across the engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_auc(scores: pd.Series, labels: pd.Series) -> float:
    """ROC AUC of `scores` for predicting a 0/1 `labels` series.

    Computed from ranks (the Mann-Whitney U identity), so it needs no
    scikit-learn and handles ties correctly. Rows where either value is
    missing are dropped. Returns NaN when only one class is present.

    0.5 means no signal; 1.0 means the score perfectly ranks positives above
    negatives; 0.0 means it perfectly ranks them below (equally informative,
    just reversed).
    """
    frame = pd.DataFrame({"s": scores, "y": labels}).dropna()
    positives = frame["y"] == 1
    n_pos = int(positives.sum())
    n_neg = len(frame) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = frame["s"].rank(method="average")
    u = ranks[positives].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auc_strength(auc: float) -> float:
    """Distance from no-signal, direction-free: 0.0 = useless, 0.5 = perfect."""
    if np.isnan(auc):
        return 0.0
    return abs(auc - 0.5)
