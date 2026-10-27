"""Memory-management benchmark harness.

Drives AMLN (V2) and the baselines over the *same* event stream using
prequential (interleaved test-then-train) evaluation -- the standard protocol
for online learning: for each event we first PREDICT with current memory
(scored against the true label), then LET the method OBSERVE it. This measures
how well memory supports detection as it is being built, which is exactly the
streaming-IDS setting.

Metrics collected per method:
  * accuracy / macro-F1 (prequential)
  * attack recall & zero-day recall (detection quality where it matters)
  * memory footprint in KB (retained items x bytes/item)
  * compression ratio (events seen / items retained)
  * peak items retained (bound check)
  * mean update latency (ms/event) and query latency (ms/event)

Results are returned as a list of dicts and can be written to CSV + plotted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.memory.adaptive_v2 import AdaptiveMemoryLandscapeV2, LandscapeV2Config
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming import StreamReplayer


@dataclass
class MethodResult:
    name: str
    accuracy: float
    macro_f1: float
    attack_recall: float
    zero_day_recall: float
    footprint_kb: float
    compression: float
    peak_items: int
    final_items: int
    update_ms: float
    query_ms: float
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "method": self.name,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "attack_recall": round(self.attack_recall, 4),
            "zero_day_recall": round(self.zero_day_recall, 4),
            "footprint_kb": round(self.footprint_kb, 2),
            "compression": round(self.compression, 1),
            "peak_items": self.peak_items,
            "final_items": self.final_items,
            "update_ms": round(self.update_ms, 4),
            "query_ms": round(self.query_ms, 4),
        }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    f1s = []
    for c in labels:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


class BenchmarkHarness:
    """Runs prequential evaluation for AMLN + baselines on one stream."""

    def __init__(self, model: BehaviorAutoencoder, replayer: StreamReplayer,
                 cfg: AMLNConfig, latent_dim: int, benign_label: int = 0,
                 zero_day_labels: set[int] | None = None,
                 bytes_per_item: int | None = None) -> None:
        self.model = model
        self.replayer = replayer
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.benign_label = benign_label
        self.zero_day_labels = zero_day_labels or set()
        # Each retained item is a latent float32 vector + an int label.
        self.bytes_per_item = bytes_per_item or (latent_dim * 4 + 8)

        # Pre-compute embeddings + recon errors once (shared across methods).
        self._embeddings, self._recon = self._encode_stream()

    def _encode_stream(self) -> tuple[np.ndarray, np.ndarray]:
        # Standardize the whole stream in one vectorized call (not row-by-row).
        feats = self.replayer.scaler.transform(self.replayer.features).astype(np.float32)
        embs, recons = [], []
        bs = self.cfg.train.batch_size
        with torch.no_grad():
            for s in range(0, len(feats), bs):
                chunk = torch.from_numpy(feats[s:s + bs])
                z, recon = self.model(chunk)
                embs.append(z.numpy())
                recons.append(torch.mean((recon - chunk) ** 2, dim=1).numpy())
        return np.concatenate(embs), np.concatenate(recons)

    # -- AMLN V2 ----------------------------------------------------------
    def run_amln(self, cfg2: LandscapeV2Config | None = None,
                 skip_warmup_eval: bool = True) -> MethodResult:
        land = AdaptiveMemoryLandscapeV2(self.latent_dim, self.cfg.landscape,
                                         cfg2 or LandscapeV2Config())
        y_true, y_pred, zd_flag = [], [], []
        anomaly_flags = []            # AMLN's "is this novel?" decision
        peak = 0
        t_update = t_query = 0.0
        labels = self.replayer.labels
        warmup = self.replayer.warmup

        for i in range(len(self._embeddings)):
            emb = self._embeddings[i]
            # test-then-train
            t0 = time.perf_counter()
            node, _ = land.retrieve(emb, hills_only=False)
            t_query += time.perf_counter() - t0
            pred = int(node.label) if node is not None else self.benign_label

            t0 = time.perf_counter()
            res = land.observe(emb, float(self.replayer.severity[i]), int(labels[i]),
                               recon_error=float(self._recon[i]))
            t_update += time.perf_counter() - t0

            if not (skip_warmup_eval and i < warmup):
                y_true.append(int(labels[i]))
                y_pred.append(pred)
                zd_flag.append(int(labels[i]) in self.zero_day_labels)
                anomaly_flags.append(bool(res.is_anomaly))
            peak = max(peak, len(land))

        return self._score("AMLN-V2", y_true, y_pred, zd_flag, peak, len(land),
                           t_update, t_query, anomaly_flags=anomaly_flags, extra={
                               "spawned": land.total_spawned,
                               "pruned": land.total_pruned,
                               "splits": land.total_splits,
                               "final_energy": round(land.total_energy(), 1),
                           })

    # -- baselines --------------------------------------------------------
    def run_baseline(self, name: str, memory) -> MethodResult:
        y_true, y_pred, zd_flag = [], [], []
        peak = 0
        t_update = t_query = 0.0
        labels = self.replayer.labels
        warmup = self.replayer.warmup

        for i in range(len(self._embeddings)):
            emb = self._embeddings[i]
            t0 = time.perf_counter()
            pred = memory.predict(emb)
            t_query += time.perf_counter() - t0

            if i >= warmup:
                y_true.append(int(labels[i]))
                y_pred.append(int(pred))
                zd_flag.append(int(labels[i]) in self.zero_day_labels)

            t0 = time.perf_counter()
            memory.observe(emb, float(self.replayer.severity[i]), int(labels[i]))
            t_update += time.perf_counter() - t0
            peak = max(peak, memory.num_items())

        return self._score(name, y_true, y_pred, zd_flag, peak,
                           memory.num_items(), t_update, t_query)

    # -- scoring ----------------------------------------------------------
    def _score(self, name, y_true, y_pred, zd_flag, peak, final_items,
               t_update, t_query, anomaly_flags=None, extra=None) -> MethodResult:
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        n = max(1, len(yt))
        acc = float(np.mean(yt == yp)) if len(yt) else 0.0
        f1 = _macro_f1(yt, yp) if len(yt) else 0.0

        attack_mask = yt != self.benign_label
        attack_recall = (float(np.mean(yp[attack_mask] == yt[attack_mask]))
                         if np.any(attack_mask) else 0.0)
        zd = np.asarray(zd_flag, dtype=bool)
        if anomaly_flags is not None:
            # AMLN detects a zero-day as "flagged novel" (surprise-gated), the
            # right definition for an unseen family it cannot name. A baseline
            # with no novelty signal can only match the exact label (near-0).
            af = np.asarray(anomaly_flags, dtype=bool)
            zero_day_recall = (float(np.mean(af[zd])) if np.any(zd) else 0.0)
        else:
            zero_day_recall = (float(np.mean(yp[zd] == yt[zd]))
                               if np.any(zd) else 0.0)

        return MethodResult(
            name=name,
            accuracy=acc,
            macro_f1=f1,
            attack_recall=attack_recall,
            zero_day_recall=zero_day_recall,
            footprint_kb=peak * self.bytes_per_item / 1024.0,
            compression=len(self._embeddings) / max(1, final_items),
            peak_items=peak,
            final_items=final_items,
            update_ms=1000.0 * t_update / len(self._embeddings),
            query_ms=1000.0 * t_query / len(self._embeddings),
            extra=extra or {},
        )

    # -- orchestration ----------------------------------------------------
    def run_all(self, window: int = 2000, reservoir_k: int = 2000,
                kmeans_k: int = 16) -> list[MethodResult]:
        from amln.benchmark.baselines import (
            StoreAllMemory, SlidingWindowMemory, ReservoirMemory, OnlineKMeansMemory)

        d = self.latent_dim
        results = [self.run_amln()]
        results.append(self.run_baseline("StoreAll", StoreAllMemory(dim=d)))
        results.append(self.run_baseline(f"SlidingWindow({window})",
                                         SlidingWindowMemory(window, dim=d)))
        results.append(self.run_baseline(f"Reservoir({reservoir_k})",
                                         ReservoirMemory(reservoir_k, dim=d)))
        results.append(self.run_baseline(f"OnlineKMeans({kmeans_k})",
                                         OnlineKMeansMemory(kmeans_k)))
        return results
