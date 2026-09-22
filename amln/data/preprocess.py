"""Feature preprocessing utilities.

Real telemetry (CIC-IDS2017, UNSW-NB15, EMBER, CIC-MalMem, ...) arrives with
heterogeneous scales. AMLN's encoder and the cosine-similarity based landscape
both assume roughly standardized inputs, so we fit a simple standard scaler on
the training split and reuse it everywhere.
"""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """Minimal, dependency-light standard scaler (mean/std normalization)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fit before transform().")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
