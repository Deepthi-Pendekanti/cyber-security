"""Closed-loop Adaptive Learning.

After the landscape is populated and predictions are made, outcomes are
observed and fed back to refine:
  - Memory significance scores
  - Reinforcement strength
  - Similarity thresholds (adaptive, per-node)
  - Predictive influence weights
  - Graph edge weights

This implements the patent's "closed-loop feedback mechanism" that enables
continual improvement without retraining the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amln.config import AMLNConfig, LandscapeConfig
from amln.memory.graph import CyberMemoryGraph
from amln.memory.landscape import AdaptiveMemoryLandscape, MemoryNode, RegionType


@dataclass
class FeedbackEvent:
    """An observed outcome that can refine the memory landscape."""
    node_idx: int                    # which memory node was involved
    predicted_label: int             # what the system predicted
    actual_label: int                # ground truth
    predicted_severity: float        # predicted threat severity
    actual_severity: float           # observed severity
    correct: bool = False            # was the prediction correct?
    # Attack chain feedback
    chain_predicted: list[int] = field(default_factory=list)
    chain_actual: list[int] = field(default_factory=list)


@dataclass
class FeedbackStats:
    """Accumulated statistics from the feedback loop."""
    total_feedback: int = 0
    correct_predictions: int = 0
    incorrect_predictions: int = 0
    severity_mse: float = 0.0
    reinforcements_applied: int = 0
    decays_applied: int = 0
    threshold_adjustments: int = 0


class AdaptiveFeedbackLoop:
    """Closed-loop learning: refine landscape + graph based on outcomes.

    Does NOT retrain the encoder. Instead, adjusts the memory dynamics:
    - Correct predictions -> reinforce the involved memory node
    - Incorrect predictions -> decay the node and adjust thresholds
    - Severity mis-estimation -> adjust node severity running mean
    - Chain accuracy -> reinforce or weaken graph edges
    """

    def __init__(self, landscape: AdaptiveMemoryLandscape,
                 graph: CyberMemoryGraph,
                 cfg: AMLNConfig) -> None:
        self.landscape = landscape
        self.graph = graph
        self.cfg = cfg
        self.stats = FeedbackStats()
        # Adaptive threshold per node (starts at the global match_similarity)
        self._node_thresholds: dict[int, float] = {}
        # Learning rate for feedback-driven adjustments
        self.lr = 0.05

    def process_feedback(self, event: FeedbackEvent) -> None:
        """Process a single outcome and adjust the landscape accordingly."""
        self.stats.total_feedback += 1
        event.correct = (event.predicted_label == event.actual_label)

        if event.correct:
            self._reinforce_correct(event)
            self.stats.correct_predictions += 1
        else:
            self._penalize_incorrect(event)
            self.stats.incorrect_predictions += 1

        # Always update severity tracking
        self._update_severity(event)

        # Update graph edges based on chain accuracy
        if event.chain_predicted and event.chain_actual:
            self._update_graph_edges(event)

        # Running MSE for severity predictions
        sev_err = (event.predicted_severity - event.actual_severity) ** 2
        n = self.stats.total_feedback
        self.stats.severity_mse += (sev_err - self.stats.severity_mse) / n

    def process_batch(self, events: list[FeedbackEvent]) -> FeedbackStats:
        """Process a batch of feedback events."""
        for event in events:
            self.process_feedback(event)
        return self.stats

    def _reinforce_correct(self, event: FeedbackEvent) -> None:
        """Reward a node for making a correct prediction."""
        if event.node_idx >= len(self.landscape.nodes):
            return
        node = self.landscape.nodes[event.node_idx]
        # Boost height proportional to severity (correct high-severity = big reward)
        boost = self.lr * (0.5 + event.actual_severity)
        node.height += boost
        self.stats.reinforcements_applied += 1

    def _penalize_incorrect(self, event: FeedbackEvent) -> None:
        """Weaken a node that made an incorrect prediction."""
        if event.node_idx >= len(self.landscape.nodes):
            return
        node = self.landscape.nodes[event.node_idx]
        # Reduce height (but don't kill it - might still be useful)
        penalty = self.lr * 0.5
        node.height = max(self.cfg.landscape.valley_threshold, node.height - penalty)
        self.stats.decays_applied += 1

        # Tighten the match threshold for this node so it's harder to match
        current_th = self._node_thresholds.get(
            event.node_idx, self.cfg.landscape.match_similarity)
        self._node_thresholds[event.node_idx] = min(0.98, current_th + 0.01)
        self.stats.threshold_adjustments += 1

    def _update_severity(self, event: FeedbackEvent) -> None:
        """Correct the node's severity running mean toward the actual value."""
        if event.node_idx >= len(self.landscape.nodes):
            return
        node = self.landscape.nodes[event.node_idx]
        # Exponential moving average update
        node.severity += self.lr * (event.actual_severity - node.severity)

    def _update_graph_edges(self, event: FeedbackEvent) -> None:
        """Reinforce or weaken graph edges based on chain prediction accuracy."""
        predicted = event.chain_predicted
        actual = event.chain_actual

        # Reinforce edges that appeared in both predicted and actual chains
        for i in range(len(predicted) - 1):
            key = (predicted[i], predicted[i + 1])
            if key in self.graph.edges:
                # Was this transition correct?
                if predicted[i + 1] in actual:
                    self.graph.edges[key].weight *= (1 + self.lr)
                else:
                    self.graph.edges[key].weight *= (1 - self.lr)
                    self.graph.edges[key].weight = max(0.01, self.graph.edges[key].weight)

        # Create edges for actual transitions not yet in the graph
        for i in range(len(actual) - 1):
            key = (actual[i], actual[i + 1])
            if key not in self.graph.edges:
                from amln.memory.graph import Edge
                self.graph.edges[key] = Edge(
                    source_idx=actual[i],
                    target_idx=actual[i + 1],
                    weight=0.5,
                    count=1,
                    mean_dt=1.0,
                )

    def get_adaptive_threshold(self, node_idx: int) -> float:
        """Get the per-node adaptive match threshold."""
        return self._node_thresholds.get(
            node_idx, self.cfg.landscape.match_similarity)

    @property
    def accuracy(self) -> float:
        total = self.stats.total_feedback
        return self.stats.correct_predictions / total if total > 0 else 0.0

    def summary(self) -> dict:
        """Return a summary of the feedback loop's state."""
        return {
            "total_feedback": self.stats.total_feedback,
            "accuracy": self.accuracy,
            "severity_mse": self.stats.severity_mse,
            "reinforcements": self.stats.reinforcements_applied,
            "decays": self.stats.decays_applied,
            "threshold_adjustments": self.stats.threshold_adjustments,
            "adapted_nodes": len(self._node_thresholds),
        }
