"""Cyber Memory Graph.

Models relationships among memory nodes (hills and slopes) as a directed
graph capturing:
  - Temporal succession (attack stage A is often followed by stage B)
  - Behavioral similarity edges (nearby nodes in latent space)
  - Co-occurrence within observation windows (events seen together in time)

The graph enables:
  - Attack chain reconstruction from partial observations
  - Threat evolution prediction (what comes next?)
  - Contextual threat intelligence linking individual behaviors to campaigns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from amln.memory.landscape import AdaptiveMemoryLandscape, MemoryNode, RegionType


@dataclass
class Edge:
    """Directed edge between two memory nodes."""
    source_idx: int
    target_idx: int
    weight: float = 1.0          # transition strength
    count: int = 1               # times this transition was observed
    mean_dt: float = 1.0         # average temporal gap (in logical ticks)

    def reinforce(self, dt: float, decay: float = 0.1) -> None:
        """Strengthen edge on repeated observation."""
        self.count += 1
        self.mean_dt += (dt - self.mean_dt) / self.count
        self.weight = self.weight * (1 - decay) + decay * np.log1p(self.count)


@dataclass
class AttackChain:
    """A reconstructed or predicted sequence of attack stages."""
    node_indices: list[int]
    node_labels: list[str]
    similarities: list[float]
    confidence: float
    predicted_next: Optional[list[int]] = None


class CyberMemoryGraph:
    """Directed graph over memory landscape nodes.

    Edges encode temporal transitions observed during continual learning.
    When event A (matching node i) is followed by event B (matching node j),
    an edge (i -> j) is created or reinforced.
    """

    def __init__(self, landscape: AdaptiveMemoryLandscape) -> None:
        self.landscape = landscape
        self.edges: dict[tuple[int, int], Edge] = {}
        self._prev_node_idx: int | None = None
        self._prev_t: float = 0.0
        self._observation_window: list[int] = []
        self._window_size: int = 10

    # -- graph construction -----------------------------------------------
    def record_transition(self, current_node_idx: int) -> None:
        """Record a temporal transition from the previous node to this one."""
        if self._prev_node_idx is not None and self._prev_node_idx != current_node_idx:
            key = (self._prev_node_idx, current_node_idx)
            dt = self.landscape._t - self._prev_t
            if key in self.edges:
                self.edges[key].reinforce(dt)
            else:
                self.edges[key] = Edge(
                    source_idx=self._prev_node_idx,
                    target_idx=current_node_idx,
                    weight=1.0,
                    count=1,
                    mean_dt=dt,
                )

        # Track co-occurrence within a sliding window
        self._observation_window.append(current_node_idx)
        if len(self._observation_window) > self._window_size:
            self._observation_window.pop(0)
        self._add_cooccurrence_edges(current_node_idx)

        self._prev_node_idx = current_node_idx
        self._prev_t = self.landscape._t

    def _add_cooccurrence_edges(self, current_idx: int) -> None:
        """Create weak edges between co-occurring nodes in the window."""
        for other_idx in self._observation_window[:-1]:
            if other_idx == current_idx:
                continue
            key = (other_idx, current_idx)
            if key not in self.edges:
                self.edges[key] = Edge(
                    source_idx=other_idx,
                    target_idx=current_idx,
                    weight=0.3,
                    count=1,
                    mean_dt=1.0,
                )

    def build_from_landscape(self, similarity_threshold: float = 0.7) -> None:
        """Add similarity-based edges between nearby memory nodes."""
        nodes = self.landscape.nodes
        n = len(nodes)
        if n < 2:
            return
        mat = self.landscape._matrix()
        for i in range(n):
            sims = mat @ mat[i]
            for j in range(i + 1, n):
                if sims[j] >= similarity_threshold:
                    # Bidirectional similarity edges (weaker than temporal)
                    for key in [(i, j), (j, i)]:
                        if key not in self.edges:
                            self.edges[key] = Edge(
                                source_idx=key[0],
                                target_idx=key[1],
                                weight=float(sims[j]) * 0.5,
                                count=1,
                                mean_dt=0.0,
                            )

    # -- graph queries ----------------------------------------------------
    def successors(self, node_idx: int, min_weight: float = 0.5
                   ) -> list[tuple[int, Edge]]:
        """Get outgoing edges from a node, sorted by weight descending."""
        out = []
        for (src, tgt), edge in self.edges.items():
            if src == node_idx and edge.weight >= min_weight:
                out.append((tgt, edge))
        out.sort(key=lambda x: x[1].weight, reverse=True)
        return out

    def predecessors(self, node_idx: int, min_weight: float = 0.5
                     ) -> list[tuple[int, Edge]]:
        """Get incoming edges to a node, sorted by weight descending."""
        inc = []
        for (src, tgt), edge in self.edges.items():
            if tgt == node_idx and edge.weight >= min_weight:
                inc.append((src, edge))
        inc.sort(key=lambda x: x[1].weight, reverse=True)
        return inc

    def find_attack_chain(self, start_node_idx: int, max_depth: int = 5,
                          min_weight: float = 0.5) -> AttackChain:
        """Trace the most likely attack chain starting from a node."""
        chain_indices = [start_node_idx]
        visited = {start_node_idx}
        current = start_node_idx

        for _ in range(max_depth):
            succs = self.successors(current, min_weight=min_weight)
            # Pick the strongest unvisited successor
            next_node = None
            for tgt_idx, edge in succs:
                if tgt_idx not in visited and tgt_idx < len(self.landscape.nodes):
                    next_node = tgt_idx
                    break
            if next_node is None:
                break
            chain_indices.append(next_node)
            visited.add(next_node)
            current = next_node

        # Compute labels and similarities along the chain
        nodes = self.landscape.nodes
        labels = []
        similarities = []
        for i, idx in enumerate(chain_indices):
            if idx < len(nodes):
                labels.append(str(nodes[idx].label))
                if i > 0 and chain_indices[i - 1] < len(nodes):
                    sim = float(nodes[idx].centroid @ nodes[chain_indices[i - 1]].centroid)
                    similarities.append(sim)
                else:
                    similarities.append(1.0)
            else:
                labels.append("unknown")
                similarities.append(0.0)

        confidence = min(1.0, np.mean([e.weight for (_, e) in
                              self.edges.items()
                              if e.source_idx in chain_indices
                              and e.target_idx in chain_indices])) if len(chain_indices) > 1 else 0.0

        # Predict next likely stages beyond the chain
        predicted = []
        if chain_indices:
            last = chain_indices[-1]
            for tgt_idx, edge in self.successors(last, min_weight=min_weight * 0.5):
                if tgt_idx not in visited and tgt_idx < len(nodes):
                    predicted.append(tgt_idx)
                    if len(predicted) >= 3:
                        break

        return AttackChain(
            node_indices=chain_indices,
            node_labels=labels,
            similarities=similarities,
            confidence=float(confidence),
            predicted_next=predicted if predicted else None,
        )

    # -- statistics -------------------------------------------------------
    def num_edges(self) -> int:
        return len(self.edges)

    def mean_edge_weight(self) -> float:
        if not self.edges:
            return 0.0
        return float(np.mean([e.weight for e in self.edges.values()]))

    def hub_nodes(self, top_k: int = 5) -> list[tuple[int, int]]:
        """Nodes with most outgoing edges (attack stage initiators)."""
        out_degree: dict[int, int] = {}
        for (src, _), _ in self.edges.items():
            out_degree[src] = out_degree.get(src, 0) + 1
        ranked = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
