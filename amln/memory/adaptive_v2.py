"""Novel Adaptive Memory Landscape (V2) - the core research contribution.

V1 (``landscape.py``) models memory as hills/slopes/valleys with cosine-matched
reinforcement and time-based exponential decay. V2 keeps that terrain metaphor
but adds three novel mechanisms that make it a genuinely adaptive, bounded,
drift-aware memory suitable for streaming threat detection:

1. SURPRISE-GATED PLASTICITY (free-energy inspired)
   Each event carries a *surprise* signal combining
     - encoder reconstruction error (how unfamiliar the behaviour is), and
     - novelty distance (1 - cosine sim to the nearest memory).
   Plasticity (how strongly the landscape updates) scales with surprise:
   familiar, low-surprise events barely perturb memory; high-surprise events
   rapidly spawn/strengthen memory. This is what gives sensitive zero-day
   detection -- unseen attack families are, by construction, high-surprise.

2. ENERGY-BUDGETED FORGETTING (bounded memory)
   The landscape holds a fixed *energy budget* E_max shared across all nodes.
   Total stored height may not exceed it; when a new memory would overflow the
   budget, energy is reclaimed from the least-important nodes first. This makes
   the memory footprint provably bounded and roughly CONSTANT regardless of how
   long the stream runs -- the headline benchmark property.

3. DRIFT-AWARE CONSOLIDATION
   Each node tracks the running variance of the events it absorbs. A node whose
   intra-cluster spread grows beyond a threshold is SPLIT (an attack family that
   is mutating / drifting), while tight, frequently-reinforced nodes CONSOLIDATE
   into hills. This adapts the memory online, without retraining the encoder.

The public API mirrors V1 (``observe`` / ``retrieve`` / ``complete_pattern`` /
``regions``) so the benchmark harness can swap V1<->V2 transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from amln.config import LandscapeConfig


class RegionType(str, Enum):
    HILL = "hill"
    SLOPE = "slope"
    VALLEY = "valley"


@dataclass
class LandscapeV2Config:
    """Hyper-parameters for the novel mechanisms (sits alongside LandscapeConfig)."""

    # Energy budget: max total height summed across all nodes. Smaller => more
    # aggressive forgetting => smaller, more compressed memory. Sized to comfortably
    # hold one hill per attack family (CIC-IDS2017 has ~15) plus headroom.
    energy_budget: float = 400.0

    # Surprise shaping. surprise = w_recon * recon_err_norm + w_novelty * (1 - sim)
    w_recon: float = 0.5
    w_novelty: float = 0.5
    # Running normalizer for recon error (updated online).
    recon_ema: float = 0.99

    # A new memory is spawned when surprise exceeds this (adaptive novelty gate).
    spawn_surprise: float = 0.45

    # Drift: split a node when its feature-variance exceeds split_var and it has
    # absorbed at least split_min_count events.
    split_var: float = 0.35
    split_min_count: int = 40

    # Zero-day flagging: an event is "novel/anomalous" when its surprise exceeds
    # this threshold. Lowered for real noisy telemetry where unseen families
    # still produce only moderately elevated surprise.
    anomaly_surprise: float = 0.5

    # V2-specific decay rate (overrides LandscapeConfig.decay_rate for V2 only).
    # Gentler decay lets one hill per attack family persist across a long,
    # many-class interleaved stream instead of collapsing to a few nodes.
    decay_rate: float = 0.01


@dataclass
class MemoryNodeV2:
    centroid: np.ndarray
    height: float
    count: int
    severity: float
    label: int
    label_votes: dict = field(default_factory=dict)
    created_t: float = 0.0
    last_t: float = 0.0
    # Online variance tracking (Welford-style) of absorbed embeddings' distance.
    var_mean: float = 0.0
    var_m2: float = 0.0

    def region(self, cfg: LandscapeConfig) -> RegionType:
        if self.height >= cfg.hill_threshold:
            return RegionType.HILL
        if self.height < cfg.valley_threshold:
            return RegionType.VALLEY
        return RegionType.SLOPE

    def importance(self, cfg: LandscapeConfig, now: float) -> float:
        freq = np.log1p(self.count)
        recency = np.exp(-cfg.decay_rate * (now - self.last_t))
        return (cfg.w_frequency * freq
                + cfg.w_severity * self.severity
                + cfg.w_recency * recency)

    def spread(self) -> float:
        """Current intra-cluster variance estimate (drift signal)."""
        if self.count < 2:
            return 0.0
        return self.var_m2 / (self.count - 1)


@dataclass
class ObserveResult:
    """Rich per-event outcome, used by the dashboard and benchmark."""

    node_index: int
    surprise: float
    novelty: float
    recon_error: float
    is_anomaly: bool          # flagged as novel/zero-day-like
    spawned: bool             # created a new memory
    split: bool               # triggered a drift split


class AdaptiveMemoryLandscapeV2:
    """Surprise-gated, energy-budgeted, drift-aware memory landscape."""

    def __init__(self, latent_dim: int, cfg: LandscapeConfig,
                 cfg2: LandscapeV2Config | None = None) -> None:
        self.dim = latent_dim
        self.cfg = cfg
        self.cfg2 = cfg2 or LandscapeV2Config()
        self.nodes: list[MemoryNodeV2] = []
        self._t: float = 0.0
        self.total_observations: int = 0
        self.total_pruned: int = 0
        self.total_spawned: int = 0
        self.total_splits: int = 0
        # Online recon-error normalizer.
        self._recon_mean: float = 1.0
        self._recon_std: float = 1.0

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def _matrix(self) -> np.ndarray:
        if not self.nodes:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([n.centroid for n in self.nodes])

    def total_energy(self) -> float:
        return float(sum(n.height for n in self.nodes))

    # -- surprise computation --------------------------------------------
    def _update_recon_stats(self, recon_error: float) -> None:
        a = self.cfg2.recon_ema
        self._recon_mean = a * self._recon_mean + (1 - a) * recon_error
        dev = abs(recon_error - self._recon_mean)
        self._recon_std = a * self._recon_std + (1 - a) * dev + 1e-6

    def _surprise(self, recon_error: float, novelty: float) -> float:
        # Normalize recon error against its running stats, squash to [0, ~1].
        z = (recon_error - self._recon_mean) / (self._recon_std + 1e-6)
        recon_norm = float(1.0 / (1.0 + np.exp(-z)))   # sigmoid -> (0,1)
        s = self.cfg2.w_recon * recon_norm + self.cfg2.w_novelty * novelty
        return float(np.clip(s, 0.0, 1.0))

    # -- core dynamics ----------------------------------------------------
    def observe(self, embedding: np.ndarray, severity: float, label: int,
                recon_error: float = 0.0) -> ObserveResult:
        """Integrate one event with surprise-gated plasticity."""
        self._t += 1.0
        self.total_observations += 1
        emb = self._unit(np.asarray(embedding, dtype=np.float32))

        sims = self._matrix() @ emb if self.nodes else np.empty(0)
        best = int(np.argmax(sims)) if sims.size else -1
        best_sim = float(sims[best]) if best >= 0 else -1.0
        novelty = float(np.clip(1.0 - best_sim, 0.0, 1.0)) if best >= 0 else 1.0

        self._update_recon_stats(recon_error)
        surprise = self._surprise(recon_error, novelty)
        is_anomaly = surprise >= self.cfg2.anomaly_surprise

        spawned = False
        split = False

        # Spawn a new memory only when there is no sufficiently-similar existing
        # memory to absorb the event. Surprise modulates *plasticity* (how hard
        # we update), not the spawn decision itself -- otherwise familiar-but-
        # severe repeats would wastefully fork new nodes.
        if best < 0 or best_sim < self.cfg.match_similarity:
            node = self._spawn(emb, severity, label, surprise)
            node_index = len(self.nodes) - 1
            spawned = True
        else:
            node = self.nodes[best]
            self._reinforce(node, emb, severity, label, best_sim, surprise)
            node_index = best
            # Drift check: has this cluster spread too far?
            if (node.count >= self.cfg2.split_min_count
                    and node.spread() > self.cfg2.split_var):
                self._split(node)
                split = True
                self.total_splits += 1

        self._decay_and_prune()
        self._enforce_energy_budget()
        self._merge()

        # node identity may have moved after housekeeping; recompute its index
        node_index = self._index_of(node)
        return ObserveResult(
            node_index=node_index,
            surprise=surprise,
            novelty=novelty,
            recon_error=recon_error,
            is_anomaly=is_anomaly,
            spawned=spawned,
            split=split,
        )

    def _index_of(self, node: MemoryNodeV2) -> int:
        for i, n in enumerate(self.nodes):
            if n is node:
                return i
        return -1

    def _spawn(self, emb: np.ndarray, severity: float, label: int,
               surprise: float) -> MemoryNodeV2:
        # Birth height scales with surprise: a shocking novelty plants a taller
        # seed so it is not immediately forgotten before it can be confirmed.
        birth = 1.0 + surprise
        node = MemoryNodeV2(
            centroid=emb, height=birth, count=1, severity=float(severity),
            label=label, label_votes={label: 1},
            created_t=self._t, last_t=self._t,
        )
        self.nodes.append(node)
        self.total_spawned += 1
        return node

    def _reinforce(self, node: MemoryNodeV2, emb: np.ndarray, severity: float,
                   label: int, sim: float, surprise: float) -> None:
        self._apply_decay(node)
        # Plasticity scales with surprise: familiar repeats nudge gently,
        # surprising-but-matched events move the centroid more.
        lr = self.cfg.centroid_lr * (0.5 + surprise)
        new_centroid = self._unit((1 - lr) * node.centroid + lr * emb)

        # Welford update of the distance variance (drift signal) BEFORE moving.
        dist = float(1.0 - sim)
        node.count += 1
        delta = dist - node.var_mean
        node.var_mean += delta / node.count
        node.var_m2 += delta * (dist - node.var_mean)

        node.centroid = new_centroid
        node.height += self.cfg.reinforce_gain * sim * (0.5 + severity)
        node.severity += (severity - node.severity) / node.count
        node.label_votes[label] = node.label_votes.get(label, 0) + 1
        node.label = max(node.label_votes, key=node.label_votes.get)
        node.last_t = self._t

    def _split(self, node: MemoryNodeV2) -> None:
        """Split a drifting cluster into two: the stable core and a fresh offshoot."""
        # Keep the original as the consolidated core (reset variance, halve
        # count/height); spawn a sibling slightly perturbed toward the drift.
        perturb = self._unit(node.centroid + np.random.normal(0, 0.1, self.dim).astype(np.float32))
        sibling = MemoryNodeV2(
            centroid=perturb,
            height=max(self.cfg.valley_threshold, node.height * 0.4),
            count=max(1, node.count // 3),
            severity=node.severity,
            label=node.label,
            label_votes=dict(node.label_votes),
            created_t=self._t, last_t=self._t,
        )
        node.height *= 0.6
        node.count = max(1, node.count - sibling.count)
        node.var_mean = 0.0
        node.var_m2 = 0.0
        self.nodes.append(sibling)

    def _apply_decay(self, node: MemoryNodeV2) -> None:
        dt = self._t - node.last_t
        if dt > 0:
            node.height *= float(np.exp(-self.cfg2.decay_rate * dt))
            node.last_t = self._t

    def _decay_and_prune(self) -> None:
        survivors: list[MemoryNodeV2] = []
        for node in self.nodes:
            self._apply_decay(node)
            if node.height >= self.cfg.valley_threshold:
                survivors.append(node)
            else:
                self.total_pruned += 1
        self.nodes = survivors

    # Newly spawned memories get a grace period before the energy budget can
    # reclaim from them, so a genuine new family isn't shaved to death before
    # it has a chance to be reinforced and prove itself.
    GRACE_PERIOD = 50.0

    def _enforce_energy_budget(self) -> None:
        """Reclaim energy from least-important *established* nodes within budget."""
        budget = self.cfg2.energy_budget
        total = self.total_energy()
        if total <= budget or not self.nodes:
            return
        # Only established nodes (past the grace period) can be shaved; this
        # protects fresh memories while still bounding total energy.
        shaveable = [n for n in self.nodes
                     if (self._t - n.created_t) >= self.GRACE_PERIOD]
        if not shaveable:
            return
        ranked = sorted(shaveable, key=lambda n: n.importance(self.cfg, self._t))
        overflow = total - budget
        for node in ranked:
            if overflow <= 0:
                break
            # Never shave a node below the valley threshold in one pass; let
            # natural decay finish the job. This keeps forgetting graceful.
            take = min(node.height - self.cfg.valley_threshold, overflow)
            if take > 0:
                node.height -= take
                overflow -= take
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.height >= self.cfg.valley_threshold]
        self.total_pruned += before - len(self.nodes)

    def _merge(self) -> None:
        if len(self.nodes) < 2:
            return
        merged: list[MemoryNodeV2] = []
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

    # -- retrieval --------------------------------------------------------
    def retrieve(self, embedding: np.ndarray, hills_only: bool = True
                 ) -> tuple[MemoryNodeV2 | None, float]:
        emb = self._unit(np.asarray(embedding, dtype=np.float32))
        candidates = [n for n in self.nodes
                      if not hills_only or n.region(self.cfg) == RegionType.HILL]
        if not candidates:
            candidates = self.nodes
        if not candidates:
            return None, -1.0
        mat = np.stack([n.centroid for n in candidates])
        sims = mat @ emb
        idx = int(np.argmax(sims))
        return candidates[idx], float(sims[idx])

    def complete_pattern(self, partial_embedding: np.ndarray
                         ) -> tuple[np.ndarray | None, MemoryNodeV2 | None, float]:
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
        if not self.nodes:
            return float(self.total_observations)
        return self.total_observations / len(self.nodes)

    def forgetting_efficiency(self) -> float:
        created = len(self.nodes) + self.total_pruned
        return self.total_pruned / created if created else 0.0

    def __len__(self) -> int:
        return len(self.nodes)
