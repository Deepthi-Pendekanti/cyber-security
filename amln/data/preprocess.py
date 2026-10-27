"""Feature preprocessing utilities.

Real telemetry (CIC-IDS2017, UNSW-NB15, EMBER, CIC-MalMem, ...) arrives with
heterogeneous, heavy-tailed features: flow durations, byte counts, and packet
rates span many orders of magnitude and contain extreme outliers. A plain
mean/std scaler leaves those outliers huge (e.g. +40 sigma), which makes the
autoencoder's reconstruction MSE explode and stall.

``StandardScaler`` keeps the simple mean/std behaviour (used by the synthetic
pipeline). ``RobustScaler`` adds a signed-log compression + clipping that tames
the long tails, which is what the CIC-IDS2017 stream needs for the encoder to
actually converge.
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


class RobustScaler:
    """Heavy-tail-aware scaler for real network-flow features.

    Pipeline per feature:
      1. signed log compression:  sign(x) * log1p(|x|)   -- tames multi-order
         magnitudes (bytes, durations) without discarding sign,
      2. median / IQR standardization  -- robust to remaining outliers,
      3. clip to +/- ``clip`` sigma    -- caps the extreme tails that otherwise
         dominate the reconstruction loss.

    Fitting on the (benign) warmup prefix and freezing keeps the stream causal.
    """

    def __init__(self, clip: float = 5.0) -> None:
        self.clip = clip
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @staticmethod
    def _signed_log(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, x: np.ndarray) -> "RobustScaler":
        xl = self._signed_log(x.astype(np.float64))
        med = np.median(xl, axis=0, keepdims=True)
        q75 = np.percentile(xl, 75, axis=0, keepdims=True)
        q25 = np.percentile(xl, 25, axis=0, keepdims=True)
        iqr = (q75 - q25)
        # Fall back to std where IQR collapses (near-constant features).
        iqr = np.where(iqr < 1e-6, xl.std(axis=0, keepdims=True) + 1e-6, iqr)
        self.center_ = med
        self.scale_ = iqr
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("RobustScaler must be fit before transform().")
        xl = self._signed_log(x.astype(np.float64))
        z = (xl - self.center_) / self.scale_
        z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
