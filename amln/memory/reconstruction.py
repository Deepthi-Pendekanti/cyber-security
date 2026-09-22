"""Memory Reconstruction Engine.

Goes beyond simple pattern completion (nearest-hill retrieval) to reconstruct
*complete attack sequences* from partial or fragmented observations. Uses the
Cyber Memory Graph to:

1. Identify the most relevant memory hills from partial evidence.
2. Retrieve the associated attack chain from the graph.
3. Reconstruct missing attack stages by walking graph edges.
4. Estimate adversarial objectives based on chain terminus.
5. Provide confidence scores and attribution for the reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.memory.graph import AttackChain, CyberMemoryGraph
from amln.memory.landscape import AdaptiveMemoryLandscape, MemoryNode, RegionType
from amln.models.encoder import BehaviorAutoencoder


@dataclass
class ReconstructionResult:
    """Full attack reconstruction from partial observations."""
    # Input
    partial_embeddings: np.ndarray        # (N, latent_dim) the partial evidence
    # Matched memories
    matched_nodes: list[MemoryNode]       # nearest hills/slopes for each partial
    match_similarities: list[float]       # cosine similarity to each match
    # Reconstructed latent
    completed_latents: np.ndarray         # (N, latent_dim) converged embeddings
    # Reconstructed features
    reconstructed_features: Optional[np.ndarray] = None  # (N, num_features)
    # Attack chain
    attack_chain: Optional[AttackChain] = None
    # Predicted next stages
    predicted_stages: Optional[list[int]] = None
    # Quality metrics
    reconstruction_confidence: float = 0.0
    coverage: float = 0.0                 # fraction of partials successfully matched


@dataclass
class AttributionReport:
    """Explainability report for a reconstruction decision."""
    query_embedding: np.ndarray
    matched_node: MemoryNode
    similarity: float
    node_region: str                      # hill / slope
    node_reinforcement_count: int
    node_severity: float
    contributing_factors: dict = field(default_factory=dict)
    attack_chain_context: Optional[AttackChain] = None
    explanation: str = ""


class MemoryReconstructionEngine:
    """Reconstructs complete attack patterns from fragmented observations."""

    def __init__(self, model: BehaviorAutoencoder,
                 landscape: AdaptiveMemoryLandscape,
                 graph: CyberMemoryGraph,
                 cfg: AMLNConfig) -> None:
        self.model = model
        self.landscape = landscape
        self.graph = graph
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)

    def reconstruct(self, partial_features: np.ndarray,
                    decode: bool = True) -> ReconstructionResult:
        """Reconstruct full behavioral patterns from partial observations.

        Args:
            partial_features: (N, num_features) partially observed events
                (missing dims should be 0 in standardized space).
            decode: whether to decode latents back to feature space.
        """
        # Step 1: Encode partial observations
        partial_emb = self._encode(partial_features)

        # Step 2: Match each partial to the nearest memory
        matched_nodes = []
        similarities = []
        completed_latents = []
        node_indices = []

        for emb in partial_emb:
            latent, node, sim = self.landscape.complete_pattern(emb)
            if node is not None:
                matched_nodes.append(node)
                similarities.append(sim)
                completed_latents.append(latent)
                # Find the index of this node in the landscape
                idx = self._find_node_index(node)
                node_indices.append(idx)
            else:
                # No match found; use the raw partial embedding
                matched_nodes.append(None)
                similarities.append(-1.0)
                completed_latents.append(emb)
                node_indices.append(-1)

        completed = np.stack(completed_latents)
        coverage = sum(1 for s in similarities if s > 0) / max(1, len(similarities))

        # Step 3: Reconstruct attack chain from the matched nodes
        attack_chain = None
        predicted_stages = None
        valid_indices = [idx for idx in node_indices if idx >= 0]
        if valid_indices:
            # Start chain from the most connected matched node
            start_idx = self._pick_chain_start(valid_indices)
            if start_idx >= 0:
                attack_chain = self.graph.find_attack_chain(
                    start_idx, max_depth=6, min_weight=0.3)
                predicted_stages = attack_chain.predicted_next

        # Step 4: Decode reconstructed latents to feature space
        reconstructed_features = None
        if decode:
            reconstructed_features = self._decode(completed)

        # Compute overall confidence
        mean_sim = float(np.mean([s for s in similarities if s > 0])) if any(
            s > 0 for s in similarities) else 0.0
        chain_conf = min(1.0, attack_chain.confidence) if attack_chain else 0.0
        reconstruction_confidence = min(1.0, 0.6 * mean_sim + 0.4 * chain_conf)

        return ReconstructionResult(
            partial_embeddings=partial_emb,
            matched_nodes=matched_nodes,
            match_similarities=similarities,
            completed_latents=completed,
            reconstructed_features=reconstructed_features,
            attack_chain=attack_chain,
            predicted_stages=predicted_stages,
            reconstruction_confidence=reconstruction_confidence,
            coverage=coverage,
        )

    def explain(self, partial_feature: np.ndarray) -> AttributionReport:
        """Generate an explainability report for a single partial observation.

        Provides the reasoning chain: which memory was matched, why, what
        attack context surrounds it, and what factors drove the decision.
        """
        emb = self._encode(partial_feature.reshape(1, -1))[0]
        latent, node, sim = self.landscape.complete_pattern(emb)

        if node is None:
            return AttributionReport(
                query_embedding=emb,
                matched_node=None,
                similarity=-1.0,
                node_region="none",
                node_reinforcement_count=0,
                node_severity=0.0,
                explanation="No matching memory found in landscape.",
            )

        region = node.region(self.cfg.landscape).value
        node_idx = self._find_node_index(node)

        # Contributing factors
        factors = {
            "cosine_similarity": float(sim),
            "node_height": float(node.height),
            "reinforcement_count": node.count,
            "severity": float(node.severity),
            "recency": float(np.exp(-self.cfg.landscape.decay_rate *
                                    (self.landscape._t - node.last_t))),
            "importance": float(node.importance(self.cfg.landscape, self.landscape._t)),
        }

        # Attack chain context
        chain = None
        if node_idx >= 0:
            chain = self.graph.find_attack_chain(node_idx, max_depth=4, min_weight=0.3)

        # Generate explanation text
        explanation = (
            f"Matched to {region} memory (label={node.label}, "
            f"severity={node.severity:.2f}) with cosine similarity {sim:.4f}. "
            f"This memory has been reinforced {node.count} times and has height "
            f"{node.height:.2f}. "
        )
        if chain and len(chain.node_indices) > 1:
            explanation += (
                f"Attack chain context: {len(chain.node_indices)} stages identified "
                f"with confidence {chain.confidence:.3f}."
            )
        if chain and chain.predicted_next:
            explanation += (
                f" Predicted next stages: {len(chain.predicted_next)} likely continuations."
            )

        return AttributionReport(
            query_embedding=emb,
            matched_node=node,
            similarity=sim,
            node_region=region,
            node_reinforcement_count=node.count,
            node_severity=node.severity,
            contributing_factors=factors,
            attack_chain_context=chain,
            explanation=explanation,
        )

    # -- internal helpers -------------------------------------------------
    def _encode(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features.astype(np.float32))
        out = []
        with torch.no_grad():
            for s in range(0, len(x), self.cfg.train.batch_size):
                chunk = x[s:s + self.cfg.train.batch_size].to(self.device)
                out.append(self.model.embed(chunk).cpu().numpy())
        return np.concatenate(out, axis=0)

    def _decode(self, latents: np.ndarray) -> np.ndarray:
        z = torch.from_numpy(latents.astype(np.float32))
        with torch.no_grad():
            recon = self.model.reconstruct(z.to(self.device)).cpu().numpy()
        return recon

    def _find_node_index(self, target_node: MemoryNode) -> int:
        """Find the index of a node in the landscape's node list."""
        for i, node in enumerate(self.landscape.nodes):
            if node is target_node:
                return i
        return -1

    def _pick_chain_start(self, candidate_indices: list[int]) -> int:
        """Pick the best node to start a chain from (most connected)."""
        best_idx = -1
        best_score = -1.0
        for idx in candidate_indices:
            succs = self.graph.successors(idx, min_weight=0.2)
            preds = self.graph.predecessors(idx, min_weight=0.2)
            score = len(succs) + len(preds) * 0.5
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx if best_idx >= 0 else (candidate_indices[0] if candidate_indices else -1)
