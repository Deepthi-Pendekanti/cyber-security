"""Step-driven live detection engine for the dashboard.

Wraps the pre-trained encoder + V2 landscape behind a single ``step()`` call
that: encodes one event, runs test-then-train, and returns a snapshot of what
happened (prediction, surprise, alert, landscape state). Keeping this logic
here means the Streamlit app only has to render snapshots -- no model code in
the UI layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.memory.adaptive_v2 import (
    AdaptiveMemoryLandscapeV2, LandscapeV2Config, RegionType)
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming.replayer import StreamReplayer, StreamEvent


@dataclass
class StepSnapshot:
    """Everything the UI needs to render one processed event."""

    index: int
    true_label: int
    true_label_name: str
    pred_label: int
    pred_label_name: str
    correct: bool
    severity: float
    surprise: float
    novelty: float
    recon_error: float
    is_anomaly: bool
    spawned: bool
    split: bool
    # Landscape state
    num_nodes: int
    hills: int
    slopes: int
    total_energy: float
    energy_budget: float
    compression: float
    # Node heights + labels for the "breathing landscape" chart
    node_heights: list = field(default_factory=list)
    node_labels: list = field(default_factory=list)
    node_counts: list = field(default_factory=list)
    node_severity: list = field(default_factory=list)
    # 2D projection (PCA) for the live latent map
    event_xy: tuple = (0.0, 0.0)
    node_xy: list = field(default_factory=list)
    # Memory-management stats
    total_spawned: int = 0
    total_pruned: int = 0
    total_splits: int = 0
    amln_kb: float = 0.0          # memory AMLN is actually using
    storeall_kb: float = 0.0      # memory a store-everything system would use
    savings_x: float = 1.0        # how many times smaller AMLN is
    # Rolling detection accuracy
    rolling_accuracy: float = 0.0


class LiveEngine:
    """Drives one event per ``step()`` through encoder + V2 landscape."""

    def __init__(self, model: BehaviorAutoencoder, replayer: StreamReplayer,
                 cfg: AMLNConfig, label_names: dict,
                 cfg2: LandscapeV2Config | None = None,
                 roll_window: int = 500) -> None:
        self.model = model
        self.replayer = replayer
        self.cfg = cfg
        self.label_names = label_names
        self.landscape = AdaptiveMemoryLandscapeV2(
            cfg.encoder.latent_dim, cfg.landscape, cfg2 or LandscapeV2Config())
        self._iter = replayer.stream()
        self._recent_correct: deque = deque(maxlen=roll_window)
        self.processed = 0
        self.alerts = 0
        # Fit a 2D PCA on warmup embeddings so events/memories can be shown on a
        # stable live latent map (components frozen, like the scaler).
        self._pca_mean, self._pca_comp = self._fit_pca()

    def _fit_pca(self):
        w = min(self.replayer.warmup, 3000)
        block = self.replayer.standardized_block(0, w)
        with torch.no_grad():
            emb = self.model.embed(torch.from_numpy(block.astype(np.float32))).numpy()
        mean = emb.mean(axis=0)
        _, _, vt = np.linalg.svd(emb - mean, full_matrices=False)
        return mean, vt[:2].T  # (dim, 2)

    def _project(self, emb: np.ndarray) -> tuple[float, float]:
        p = (emb - self._pca_mean) @ self._pca_comp
        return float(p[0]), float(p[1])

    def _name(self, label: int) -> str:
        return self.label_names.get(int(label), f"family_{label}")

    def step(self) -> StepSnapshot | None:
        """Process the next event; return a snapshot, or None when exhausted."""
        try:
            ev: StreamEvent = next(self._iter)
        except StopIteration:
            return None

        x = torch.from_numpy(ev.features[None, :].astype(np.float32))
        with torch.no_grad():
            z, recon = self.model(x)
            recon_err = float(torch.mean((recon - x) ** 2).item())
        emb = z.numpy()[0]

        # test-then-train
        node, _ = self.landscape.retrieve(emb, hills_only=False)
        pred = int(node.label) if node is not None else 0
        res = self.landscape.observe(emb, ev.severity, ev.label, recon_error=recon_err)

        correct = (pred == ev.label)
        self._recent_correct.append(1 if correct else 0)
        self.processed += 1
        if res.is_anomaly:
            self.alerts += 1

        reg = self.landscape.regions()
        heights = [round(float(n.height), 3) for n in self.landscape.nodes]
        labels = [self._name(n.label) for n in self.landscape.nodes]
        counts = [int(n.count) for n in self.landscape.nodes]
        sev = [round(float(n.severity), 3) for n in self.landscape.nodes]
        node_xy = [self._project(n.centroid) for n in self.landscape.nodes]
        event_xy = self._project(emb)

        # Memory-management accounting: each stored item is a latent vector
        # (dim x 4 bytes float32) + an 8-byte label. AMLN keeps one per memory
        # node; a store-everything system keeps one per event ever seen.
        bytes_per_item = self.cfg.encoder.latent_dim * 4 + 8
        amln_kb = len(self.landscape) * bytes_per_item / 1024.0
        storeall_kb = self.processed * bytes_per_item / 1024.0
        savings_x = (storeall_kb / amln_kb) if amln_kb > 0 else 1.0

        return StepSnapshot(
            index=ev.index,
            true_label=ev.label,
            true_label_name=self._name(ev.label),
            pred_label=pred,
            pred_label_name=self._name(pred),
            correct=correct,
            severity=ev.severity,
            surprise=res.surprise,
            novelty=res.novelty,
            recon_error=recon_err,
            is_anomaly=res.is_anomaly,
            spawned=res.spawned,
            split=res.split,
            num_nodes=len(self.landscape),
            hills=reg[RegionType.HILL],
            slopes=reg[RegionType.SLOPE],
            total_energy=round(self.landscape.total_energy(), 2),
            energy_budget=self.landscape.cfg2.energy_budget,
            compression=round(self.landscape.compression_ratio(), 1),
            node_heights=heights,
            node_labels=labels,
            node_counts=counts,
            node_severity=sev,
            event_xy=event_xy,
            node_xy=node_xy,
            total_spawned=self.landscape.total_spawned,
            total_pruned=self.landscape.total_pruned,
            total_splits=self.landscape.total_splits,
            amln_kb=round(amln_kb, 2),
            storeall_kb=round(storeall_kb, 2),
            savings_x=round(savings_x, 1),
            rolling_accuracy=(sum(self._recent_correct) / len(self._recent_correct)
                              if self._recent_correct else 0.0),
        )
