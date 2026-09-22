"""Metrics for AMLN.

Beyond conventional classification scores, AMLN is judged on memory behavior:
compression, retrieval accuracy, adaptive forgetting, pattern reconstruction,
storage reduction, retrieval latency, and zero-day detection.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from amln.config import AMLNConfig
from amln.data import ATTACK_FAMILIES
from amln.memory.landscape import AdaptiveMemoryLandscape
from amln.models.encoder import BehaviorAutoencoder


def _embed(model: BehaviorAutoencoder, features: np.ndarray,
           cfg: AMLNConfig) -> np.ndarray:
    device = torch.device(cfg.train.device)
    x = torch.from_numpy(features.astype(np.float32))
    out = []
    with torch.no_grad():
        for s in range(0, len(x), cfg.train.batch_size):
            out.append(model.embed(x[s:s + cfg.train.batch_size].to(device))
                       .cpu().numpy())
    return np.concatenate(out, axis=0)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Standard accuracy / precision / recall / F1 (macro-averaged)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro",
                                     zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def retrieval_accuracy(model: BehaviorAutoencoder,
                       landscape: AdaptiveMemoryLandscape,
                       features: np.ndarray, labels: np.ndarray,
                       cfg: AMLNConfig) -> dict:
    """Classify each event by its nearest memory and score label agreement.

    Also reports mean retrieval latency in milliseconds.
    """
    embeddings = _embed(model, features, cfg)
    preds = np.empty(len(labels), dtype=np.int64)

    # Classify by the nearest learned memory (hill or slope): both hills
    # (long-term) and slopes (short-term) encode valid behavioral knowledge.
    t0 = time.perf_counter()
    for i, emb in enumerate(embeddings):
        node, _ = landscape.retrieve(emb, hills_only=False)
        preds[i] = node.label if node is not None else 0
    elapsed = time.perf_counter() - t0

    report = classification_report(labels, preds)
    report["retrieval_latency_ms"] = 1000.0 * elapsed / max(1, len(labels))
    report["predictions"] = preds
    return report


def reconstruction_accuracy(model: BehaviorAutoencoder,
                            landscape: AdaptiveMemoryLandscape,
                            partial_features: np.ndarray,
                            full_features: np.ndarray,
                            cfg: AMLNConfig) -> dict:
    """Behavior-driven pattern completion quality.

    Encode partial events, converge each to its nearest hill, decode the
    completed latent, and compare the reconstructed feature vector to the true
    full event via cosine similarity.
    """
    partial_emb = _embed(model, partial_features, cfg)
    device = torch.device(cfg.train.device)

    sims, matched = [], 0
    completed_latents = []
    for emb in partial_emb:
        latent, node, _ = landscape.complete_pattern(emb)
        if latent is None:
            completed_latents.append(emb)  # nothing learned yet; passthrough
        else:
            completed_latents.append(latent)
            matched += 1

    latents = torch.from_numpy(np.stack(completed_latents).astype(np.float32))
    with torch.no_grad():
        recon = model.reconstruct(latents.to(device)).cpu().numpy()

    for r, full in zip(recon, full_features):
        sims.append(_cosine_np(r, full))

    return {
        "pattern_reconstruction_cosine": float(np.mean(sims)),
        "match_rate": matched / len(partial_features),
    }


def zero_day_detection(model: BehaviorAutoencoder,
                       landscape: AdaptiveMemoryLandscape,
                       features: np.ndarray, labels: np.ndarray,
                       cfg: AMLNConfig, novelty_threshold: float = 0.9) -> dict:
    """Detection performance on held-out (unseen) attack families.

    An event is flagged as a threat if either (a) it converges to a non-benign
    hill, or (b) its similarity to the nearest hill is below ``novelty_threshold``
    (novel behavior). Detection rate = fraction of true attacks flagged.
    """
    embeddings = _embed(model, features, cfg)
    benign_idx = ATTACK_FAMILIES.index("benign")

    detected = 0
    total_attacks = 0
    for emb, lab in zip(embeddings, labels):
        is_attack = lab != benign_idx
        node, sim = landscape.retrieve(emb, hills_only=True)
        flagged = node is not None and (node.label != benign_idx
                                        or sim < novelty_threshold)
        if is_attack:
            total_attacks += 1
            detected += int(flagged)

    return {
        "zero_day_detection_rate": detected / total_attacks if total_attacks else 0.0,
        "evaluated_attacks": total_attacks,
    }


def memory_footprint(landscape: AdaptiveMemoryLandscape, latent_dim: int,
                     num_raw_features: int) -> dict:
    """Storage reduction vs. keeping every raw event.

    Compares bytes to store all observed raw feature vectors against bytes to
    store the retained memory-node centroids.
    """
    bytes_per_float = 4
    raw_bytes = landscape.total_observations * num_raw_features * bytes_per_float
    mem_bytes = len(landscape) * latent_dim * bytes_per_float
    reduction = 1.0 - (mem_bytes / raw_bytes) if raw_bytes else 0.0
    return {
        "memory_nodes": len(landscape),
        "compression_ratio": landscape.compression_ratio(),
        "forgetting_efficiency": landscape.forgetting_efficiency(),
        "storage_reduction": reduction,
        "raw_kb": raw_bytes / 1024.0,
        "memory_kb": mem_bytes / 1024.0,
    }


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
