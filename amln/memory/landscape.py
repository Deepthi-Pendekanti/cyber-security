"""Adaptive Memory Landscape.

This module models memory as a dynamic terrain over the 64-d latent space
rather than as a flat database of embeddings. Each memory is a *node* with a
centroid and a scalar *height* (stability). The height evolves under four
biologically inspired forces:

* **Reinforcement** - repeated, similar, high-severity observations raise a
  node's height, growing slopes into stable hills (long-term memory).
* **Decay** - height leaks away exponentially with elapsed time, so
  un-reinforced memories sink toward valleys (natural forgetting).
* **Merging** - nodes whose centroids become very similar collapse into one,
  modelling related attack families converging into a single hill.
* **Pruning** - nodes that fall below the valley threshold are removed,
  bounding memory growth.

Region classification by height:
    height >= hill_threshold      -> HILL   (long-term knowledge)
    valley_threshold <= h < hill  -> SLOPE  (short-term memory)
    height <  valley_threshold    -> VALLEY (forgotten; pruned)

Retrieval and pattern completion both work by finding the nearest hill to a
(possibly partial) query embedding - analogous to recalling a whole memory
from a fragment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from amln.config import LandscapeConfig


class RegionType(str, Enum):
    HILL = "hill"      # stable, long-term memory
    SLOPE = "slope"    # short-term memory
    VALLEY = "valley"  # decayed / forgotten


@dataclass
class MemoryNode:
    """A single dynamic region of the memory landscape."""

    centroid: np.ndarray            # (latent_dim,) unit-norm behavioral prototype
    height: float                   # stability / long-term importance
    count: int                      # times reinforced (frequency)
    severity: float                 # running mean severity of matched events
    label: int                      # dominant attack-family label
    label_votes: dict = field(default_factory=dict)
    created_t: float = 0.0
    last_t: float = 0.0             # time of last reinforcement (for decay)

    def region(self, cfg: LandscapeConfig) -> RegionType:
        if self.height >= cfg.hill_threshold:
            return RegionType.HILL
        if self.height < cfg.valley_threshold:
            return RegionType.VALLEY
        return RegionType.SLOPE

    def importance(self, cfg: LandscapeConfig, now: float) -> float:
        """Composite importance from frequency, severity, and recency."""
        freq = np.log1p(self.count)
        recency = np.exp(-cfg.decay_rate * (now - self.last_t))
        return (cfg.w_frequency * freq
                + cfg.w_severity * self.severity
                + cfg.w_recency * recency)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of unit-norm vector ``a`` against rows of ``b``."""
    if b.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    return b @ a


class AdaptiveMemoryLandscape:
    """Continual, incremental memory over 64-d behavioral embeddings."""

    def __init__(self, latent_dim: int, cfg: LandscapeConfig) -> None:
        self.dim = latent_dim
        self.cfg = cfg
        self.nodes: list[MemoryNode] = []
        self._t: float = 0.0                 # logical clock (increments per observe)
        self.total_observations: int = 0
        self.total_pruned: int = 0

    # -- internal helpers -------------------------------------------------
    def _matrix(self) -> np.ndarray:
        if not self.nodes:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([n.centroid for n in self.nodes])

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    # -- core dynamics ----------------------------------------------------
    def observe(self, embedding: np.ndarray, severity: float, label: int) -> MemoryNode:
        """Integrate one new event; reinforce the nearest match or spawn a slope."""
        self._t += 1.0
        self.total_observations += 1
        emb = self._unit(np.asarray(embedding, dtype=np.float32))

        sims = _cosine(emb, self._matrix())
        best = int(np.argmax(sims)) if sims.size else -1

        if best >= 0 and sims[best] >= self.cfg.match_similarity:
            node = self._reinforce(self.nodes[best], emb, severity, label, sims[best])
        else:
            node = self._spawn(emb, severity, label)

        # Housekeeping: forget stale memories and merge converging families.
        self._decay_and_prune()
        self._merge()
        return node

    def _spawn(self, emb: np.ndarray, severity: float, label: int) -> MemoryNode:
        node = MemoryNode(
            centroid=emb,
            height=1.0,                      # born as a slope
            count=1,
            severity=float(severity),
            label=label,
            label_votes={label: 1},
            created_t=self._t,
            last_t=self._t,
        )
        self.nodes.append(node)
        return node

    def _reinforce(self, node: MemoryNode, emb: np.ndarray, severity: float,
                   label: int, sim: float) -> MemoryNode:
        # Apply pending decay up to now before adding new strength.
        self._apply_decay(node)

        # Weighted online centroid update toward the new observation.
        lr = self.cfg.centroid_lr
        node.centroid = self._unit((1 - lr) * node.centroid + lr * emb)

        # Height gain scales with match quality and severity: severe, textbook
        # repeats of a known attack build hills fastest.
        node.height += self.cfg.reinforce_gain * sim * (0.5 + severity)
        node.count += 1
        node.severity += (severity - node.severity) / node.count  # running mean
        node.label_votes[label] = node.label_votes.get(label, 0) + 1
        node.label = max(node.label_votes, key=node.label_votes.get)
        node.last_t = self._t
        return node

    def _apply_decay(self, node: MemoryNode) -> None:
        dt = self._t - node.last_t
        if dt > 0:
            node.height *= float(np.exp(-self.cfg.decay_rate * dt))
            node.last_t = self._t

    def _decay_and_prune(self) -> None:
        survivors: list[MemoryNode] = []
        for node in self.nodes:
            self._apply_decay(node)
            if node.height >= self.cfg.valley_threshold:
                survivors.append(node)
            else:
                self.total_pruned += 1
        self.nodes = survivors

    def _merge(self) -> None:
        """Collapse pairs of nodes whose centroids have converged."""
        if len(self.nodes) < 2:
            return
        merged: list[MemoryNode] = []
        consumed = set()
        for i, node in enumerate(self.nodes):
            if i in consumed:
                continue
            for j in range(i + 1, len(self.nodes)):
                if j in consumed:
                    continue
                sim = float(node.centroid @ self.nodes[j].centroid)
                if sim >= self.cfg.merge_similarity:
                    other = self.nodes[j]
                    total = node.count + other.count
                    # Height-weighted centroid, summed height (families join).
                    node.centroid = self._unit(
                        node.height * node.centroid + other.height * other.centroid)
                    node.severity = (node.severity * node.count
                                     + other.severity * other.count) / total
                    node.height = max(node.height, other.height) + 0.5 * min(
                        node.height, other.height)
                    node.count = total
                    for lab, v in other.label_votes.items():
                        node.label_votes[lab] = node.label_votes.get(lab, 0) + v
                    node.label = max(node.label_votes, key=node.label_votes.get)
                    consumed.add(j)
            merged.append(node)
        self.nodes = merged

    # -- retrieval / pattern completion ----------------------------------
    def retrieve(self, embedding: np.ndarray, hills_only: bool = True
                 ) -> tuple[MemoryNode | None, float]:
        """Return the nearest memory (optionally restricted to hills) + similarity."""
        emb = self._unit(np.asarray(embedding, dtype=np.float32))
        candidates = [n for n in self.nodes
                      if not hills_only or n.region(self.cfg) == RegionType.HILL]
        if not candidates:
            # Fall back to any memory when no hill has formed yet.
            candidates = self.nodes
        if not candidates:
            return None, -1.0
        mat = np.stack([n.centroid for n in candidates])
        sims = mat @ emb
        idx = int(np.argmax(sims))
        return candidates[idx], float(sims[idx])

    def complete_pattern(self, partial_embedding: np.ndarray
                         ) -> tuple[np.ndarray | None, MemoryNode | None, float]:
        """Converge a partial embedding to its nearest hill centroid.

        Returns the reconstructed (completed) latent vector, the matched node,
        and the similarity. The completed latent can be fed to the decoder to
        recover the full behavioral pattern before the attack fully unfolds.
        """
        # Prefer converging onto a stable hill, but fall back to the nearest
        # learned prototype so partial events of not-yet-stabilized families
        # can still be completed.
        node, sim = self.retrieve(partial_embedding, hills_only=True)
        alt, alt_sim = self.retrieve(partial_embedding, hills_only=False)
        if alt is not None and alt_sim > sim:
            node, sim = alt, alt_sim
        if node is None:
            return None, None, -1.0
        return node.centroid.copy(), node, sim

    # -- introspection ----------------------------------------------------
    def regions(self) -> dict[RegionType, int]:
        counts = {r: 0 for r in RegionType}
        for n in self.nodes:
            counts[n.region(self.cfg)] += 1
        return counts

    def compression_ratio(self) -> float:
        """Observations absorbed per retained memory node (higher = more compact)."""
        if not self.nodes:
            return float(self.total_observations)
        return self.total_observations / len(self.nodes)

    def forgetting_efficiency(self) -> float:
        """Fraction of created memories that were later forgotten (pruned)."""
        created = len(self.nodes) + self.total_pruned
        return self.total_pruned / created if created else 0.0

    def __len__(self) -> int:
        return len(self.nodes)
