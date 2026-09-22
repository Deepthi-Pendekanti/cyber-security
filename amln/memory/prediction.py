"""Threat Evolution Prediction Engine.

Forecasts probable future attack behaviors by analyzing:
  - Temporal transition patterns in the Cyber Memory Graph
  - Historical behavioral trajectories stored in memory hills
  - Attack stage progression sequences
  - Memory reinforcement trends

Provides predictions at multiple levels:
  - Next-stage prediction: what single behavior is most likely next?
  - Multi-step trajectory: full predicted attack progression
  - Threat escalation scoring: will this behavior chain escalate?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from amln.config import AMLNConfig
from amln.memory.graph import CyberMemoryGraph, Edge
from amln.memory.landscape import AdaptiveMemoryLandscape, MemoryNode, RegionType


@dataclass
class ThreatPrediction:
    """A predicted future attack stage."""
    node_idx: int
    node_label: int
    probability: float              # estimated transition probability
    severity: float                 # severity of the predicted stage
    expected_dt: float              # expected time until this stage
    confidence: float               # overall confidence in prediction


@dataclass
class ThreatTrajectory:
    """A multi-step predicted attack trajectory."""
    stages: list[ThreatPrediction]
    total_confidence: float
    escalation_score: float         # 0-1, how much severity escalates
    campaign_label: str             # dominant attack family
    trajectory_length: int


@dataclass
class ThreatEvolutionReport:
    """Complete prediction output for a given current state."""
    current_node_idx: int
    current_node_label: int
    current_severity: float
    # Single-step predictions
    next_stages: list[ThreatPrediction]
    # Multi-step trajectory
    trajectory: ThreatTrajectory
    # Risk assessment
    escalation_probability: float
    max_predicted_severity: float
    risk_score: float               # combined risk metric


class ThreatEvolutionEngine:
    """Predicts future attack progression from current behavioral state."""

    def __init__(self, landscape: AdaptiveMemoryLandscape,
                 graph: CyberMemoryGraph,
                 cfg: AMLNConfig) -> None:
        self.landscape = landscape
        self.graph = graph
        self.cfg = cfg

    def predict_next(self, current_node_idx: int,
                     top_k: int = 5) -> list[ThreatPrediction]:
        """Predict the most likely next attack stages from the current node."""
        successors = self.graph.successors(current_node_idx, min_weight=0.2)
        if not successors:
            return []

        # Compute transition probabilities from edge weights
        total_weight = sum(edge.weight for _, edge in successors)
        predictions = []
        for tgt_idx, edge in successors[:top_k]:
            if tgt_idx >= len(self.landscape.nodes):
                continue
            node = self.landscape.nodes[tgt_idx]
            prob = edge.weight / total_weight if total_weight > 0 else 0.0
            # Confidence combines edge strength, node stability, and count
            confidence = min(1.0, (
                0.4 * (edge.weight / max(1.0, total_weight)) +
                0.3 * min(1.0, node.height / self.cfg.landscape.hill_threshold) +
                0.3 * min(1.0, edge.count / 10.0)
            ))
            predictions.append(ThreatPrediction(
                node_idx=tgt_idx,
                node_label=node.label,
                probability=float(prob),
                severity=float(node.severity),
                expected_dt=float(edge.mean_dt),
                confidence=float(confidence),
            ))

        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions

    def predict_trajectory(self, start_node_idx: int,
                           max_steps: int = 6,
                           min_probability: float = 0.1) -> ThreatTrajectory:
        """Predict a multi-step attack trajectory (greedy most-probable path)."""
        stages: list[ThreatPrediction] = []
        visited = {start_node_idx}
        current = start_node_idx
        cumulative_confidence = 1.0

        for _ in range(max_steps):
            next_preds = self.predict_next(current, top_k=3)
            # Pick the highest-probability unvisited stage
            chosen = None
            for pred in next_preds:
                if pred.node_idx not in visited and pred.probability >= min_probability:
                    chosen = pred
                    break
            if chosen is None:
                break
            stages.append(chosen)
            visited.add(chosen.node_idx)
            cumulative_confidence *= chosen.confidence
            current = chosen.node_idx

        # Compute escalation: how much does severity increase along the trajectory?
        if stages:
            start_node = self.landscape.nodes[start_node_idx]
            start_sev = start_node.severity
            max_sev = max(s.severity for s in stages)
            escalation = max(0.0, max_sev - start_sev)
        else:
            escalation = 0.0

        # Dominant label in trajectory
        if stages:
            label_counts: dict[int, int] = {}
            for s in stages:
                label_counts[s.node_label] = label_counts.get(s.node_label, 0) + 1
            campaign_label = str(max(label_counts, key=label_counts.get))
        else:
            campaign_label = str(self.landscape.nodes[start_node_idx].label
                                 if start_node_idx < len(self.landscape.nodes) else -1)

        return ThreatTrajectory(
            stages=stages,
            total_confidence=float(cumulative_confidence),
            escalation_score=float(min(1.0, escalation)),
            campaign_label=campaign_label,
            trajectory_length=len(stages),
        )

    def assess_threat(self, current_node_idx: int) -> ThreatEvolutionReport:
        """Full threat evolution assessment for a current behavioral state."""
        if current_node_idx >= len(self.landscape.nodes):
            raise ValueError(f"Node index {current_node_idx} out of range")

        current_node = self.landscape.nodes[current_node_idx]
        next_stages = self.predict_next(current_node_idx, top_k=5)
        trajectory = self.predict_trajectory(current_node_idx, max_steps=6)

        # Risk scoring
        max_pred_sev = max((s.severity for s in next_stages), default=current_node.severity)
        escalation_prob = trajectory.escalation_score
        # Risk = combination of current severity, escalation potential, and confidence
        risk_score = (
            0.3 * current_node.severity +
            0.3 * max_pred_sev +
            0.2 * escalation_prob +
            0.2 * trajectory.total_confidence
        )

        return ThreatEvolutionReport(
            current_node_idx=current_node_idx,
            current_node_label=current_node.label,
            current_severity=float(current_node.severity),
            next_stages=next_stages,
            trajectory=trajectory,
            escalation_probability=float(escalation_prob),
            max_predicted_severity=float(max_pred_sev),
            risk_score=float(min(1.0, risk_score)),
        )

    def batch_risk_assessment(self) -> list[tuple[int, float]]:
        """Score all hill nodes by their predicted risk (for SOC prioritization)."""
        risk_scores = []
        for i, node in enumerate(self.landscape.nodes):
            if node.region(self.cfg.landscape) == RegionType.HILL:
                try:
                    report = self.assess_threat(i)
                    risk_scores.append((i, report.risk_score))
                except (ValueError, IndexError):
                    continue
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        return risk_scores
