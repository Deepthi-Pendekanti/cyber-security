"""Baseline memory strategies for the benchmark.

Every baseline exposes the same minimal streaming interface so the harness can
drive them identically to AMLN:

    observe(embedding, severity, label) -> None     # integrate one event
    predict(embedding) -> int                        # label of nearest memory
    num_items() -> int                               # retained items (footprint)

They represent the standard ways a streaming system "remembers" past events:

* StoreAll        - keep every embedding (upper bound on accuracy, unbounded memory).
* SlidingWindow   - keep only the most recent W embeddings (bounded, forgets by age).
* Reservoir       - uniform random sample of size k over the whole stream (bounded).
* OnlineKMeans    - k moving centroids (bounded, but fixed capacity & no forgetting).

The point of the benchmark is to show AMLN matches StoreAll accuracy at a memory
footprint comparable to the bounded baselines, while adapting better than them.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class _ArrayKNN:
    """Nearest-neighbour prediction over a preallocated, growable array.

    Using a contiguous float32 matrix (and a single matrix-vector product per
    query) instead of re-stacking a Python list every call is what makes the
    baselines tractable at CIC-IDS2017 scale (tens of thousands of items).
    """

    def __init__(self, capacity: int, dim: int) -> None:
        self._mat = np.zeros((capacity, dim), dtype=np.float32)
        self._lab = np.zeros(capacity, dtype=np.int64)
        self._n = 0
        self._cap = capacity
        self._dim = dim

    def _append(self, emb: np.ndarray, label: int) -> None:
        if self._n >= self._cap:  # grow (amortized O(1))
            self._mat = np.resize(self._mat, (self._cap * 2, self._dim))
            self._lab = np.resize(self._lab, self._cap * 2)
            self._cap *= 2
        self._mat[self._n] = emb
        self._lab[self._n] = label
        self._n += 1

    def predict(self, embedding: np.ndarray) -> int:
        if self._n == 0:
            return 0
        emb = _unit(np.asarray(embedding, dtype=np.float32))
        sims = self._mat[:self._n] @ emb
        return int(self._lab[int(np.argmax(sims))])

    def num_items(self) -> int:
        return self._n


class StoreAllMemory(_ArrayKNN):
    """Keep every observed embedding. Unbounded; accuracy upper bound."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(capacity=16384, dim=dim)

    def observe(self, embedding: np.ndarray, severity: float, label: int) -> None:
        self._append(_unit(np.asarray(embedding, dtype=np.float32)), int(label))


class SlidingWindowMemory(_ArrayKNN):
    """Keep only the most recent ``window`` embeddings (ring buffer)."""

    def __init__(self, window: int = 2000, dim: int = 64) -> None:
        super().__init__(capacity=window, dim=dim)
        self.window = window
        self._full = False
        self._head = 0

    def observe(self, embedding: np.ndarray, severity: float, label: int) -> None:
        emb = _unit(np.asarray(embedding, dtype=np.float32))
        self._mat[self._head] = emb
        self._lab[self._head] = int(label)
        self._head = (self._head + 1) % self.window
        if not self._full:
            self._n += 1
            if self._n >= self.window:
                self._full = True


class ReservoirMemory(_ArrayKNN):
    """Reservoir sampling: a uniform random sample of size ``k`` over the stream."""

    def __init__(self, k: int = 2000, seed: int = 42, dim: int = 64) -> None:
        super().__init__(capacity=k, dim=dim)
        self.k = k
        self.rng = np.random.default_rng(seed)
        self._seen = 0

    def observe(self, embedding: np.ndarray, severity: float, label: int) -> None:
        self._seen += 1
        emb = _unit(np.asarray(embedding, dtype=np.float32))
        if self._n < self.k:
            self._mat[self._n] = emb
            self._lab[self._n] = int(label)
            self._n += 1
        else:
            j = int(self.rng.integers(0, self._seen))
            if j < self.k:
                self._mat[j] = emb
                self._lab[j] = int(label)


class OnlineKMeansMemory:
    """``k`` moving centroids updated online; label by majority vote per centroid."""

    def __init__(self, k: int = 16, lr: float = 0.1) -> None:
        self.k = k
        self.lr = lr
        self.centroids: list[np.ndarray] = []
        self.votes: list[dict] = []

    def observe(self, embedding: np.ndarray, severity: float, label: int) -> None:
        emb = _unit(np.asarray(embedding, dtype=np.float32))
        if len(self.centroids) < self.k:
            self.centroids.append(emb.copy())
            self.votes.append({int(label): 1})
            return
        mat = np.stack(self.centroids)
        idx = int(np.argmax(mat @ emb))
        self.centroids[idx] = _unit((1 - self.lr) * self.centroids[idx] + self.lr * emb)
        self.votes[idx][int(label)] = self.votes[idx].get(int(label), 0) + 1

    def predict(self, embedding: np.ndarray) -> int:
        if not self.centroids:
            return 0
        emb = _unit(np.asarray(embedding, dtype=np.float32))
        mat = np.stack(self.centroids)
        idx = int(np.argmax(mat @ emb))
        v = self.votes[idx]
        return int(max(v, key=v.get)) if v else 0

    def num_items(self) -> int:
        return len(self.centroids)
