"""Replay a chronologically ordered dataset as a live event stream.

The replayer turns an in-memory array of events into a generator that yields
one :class:`StreamEvent` at a time, optionally throttled to a target rate so a
dashboard can animate the landscape as if events were arriving live. It is
dataset-agnostic: it accepts raw ``(features, labels, severity, timestamps)``
arrays, so it works with both the synthetic generator and the CIC-IDS2017
loader.

Causality: a scaler is fit on a *prefix* (the warmup window) and then frozen,
so later events are standardized without peeking at the future -- the honest
setup for a streaming benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from amln.data.preprocess import RobustScaler, StandardScaler


@dataclass
class StreamEvent:
    """One event emitted by the replayer."""

    index: int
    raw_features: np.ndarray      # (F,) un-standardized
    features: np.ndarray          # (F,) standardized with the frozen scaler
    label: int
    severity: float
    timestamp: float
    is_warmup: bool               # True while within the warmup/fit prefix


class StreamReplayer:
    """Replays events in order, standardizing causally against a warmup prefix."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 severity: np.ndarray, timestamps: np.ndarray | None = None,
                 warmup: int = 2000, scaler: str = "robust") -> None:
        n = len(features)
        if timestamps is None:
            timestamps = np.arange(n, dtype=np.float64)
        order = np.argsort(timestamps, kind="stable")
        self.features = np.asarray(features, dtype=np.float32)[order]
        self.labels = np.asarray(labels, dtype=np.int64)[order]
        self.severity = np.asarray(severity, dtype=np.float32)[order]
        self.timestamps = np.asarray(timestamps, dtype=np.float64)[order]
        self.warmup = min(warmup, n)

        # Fit the scaler only on the warmup prefix, then freeze it (causal).
        # "robust" handles heavy-tailed real flow features; "standard" matches
        # the synthetic pipeline.
        scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()
        self.scaler = scaler_obj.fit(self.features[:max(1, self.warmup)])

    def __len__(self) -> int:
        return len(self.features)

    def stream(self, rate_hz: float | None = None,
               start: int = 0, limit: int | None = None) -> Iterator[StreamEvent]:
        """Yield events one at a time.

        Args:
            rate_hz: if set, throttle emission to this many events/second
                (for a live, watchable dashboard). ``None`` = as fast as possible.
            start: index to begin from.
            limit: max number of events to emit.
        """
        delay = (1.0 / rate_hz) if rate_hz and rate_hz > 0 else 0.0
        end = len(self.features) if limit is None else min(len(self.features), start + limit)

        for i in range(start, end):
            raw = self.features[i]
            std = self.scaler.transform(raw[None, :])[0]
            yield StreamEvent(
                index=i,
                raw_features=raw,
                features=std,
                label=int(self.labels[i]),
                severity=float(self.severity[i]),
                timestamp=float(self.timestamps[i]),
                is_warmup=(i < self.warmup),
            )
            if delay:
                time.sleep(delay)

    def standardized_block(self, start: int, count: int) -> np.ndarray:
        """Return a standardized block of events (used for batch encoder warmup)."""
        block = self.features[start:start + count]
        return self.scaler.transform(block)
