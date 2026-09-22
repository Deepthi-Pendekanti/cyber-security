"""Synthetic cyber-event generator.

Produces structured feature vectors that stand in for preprocessed security
telemetry (process creation, PowerShell execution, registry modification,
file encryption, network connections, privilege escalation, authentication).

Each attack family occupies a distinct region of feature space with its own
covariance, so that a learned encoder can separate them and the adaptive
memory landscape can form one hill per family. A subset of families can be
held out to simulate zero-day behavior at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVENT_TYPES = [
    "process_creation",
    "powershell_exec",
    "registry_modification",
    "file_encryption",
    "network_connection",
    "privilege_escalation",
    "authentication",
]

# label 0 is reserved for benign; the rest are attack families.
ATTACK_FAMILIES = [
    "benign",
    "ransomware",
    "lateral_movement",
    "privilege_escalation",
    "data_exfiltration",
    "apt_persistence",
    "credential_theft",
    "insider_threat",
]

# Rough severity per family (0..1); benign is low.
_FAMILY_SEVERITY = {
    "benign": 0.05,
    "ransomware": 0.95,
    "lateral_movement": 0.8,
    "privilege_escalation": 0.85,
    "data_exfiltration": 0.9,
    "apt_persistence": 0.88,
    "credential_theft": 0.82,
    "insider_threat": 0.6,
}


@dataclass
class CyberEventBatch:
    """A batch of generated events."""

    features: np.ndarray      # (N, num_features) float32
    labels: np.ndarray        # (N,) int  -> index into ATTACK_FAMILIES
    severity: np.ndarray      # (N,) float in [0, 1]
    event_type: np.ndarray    # (N,) int  -> index into EVENT_TYPES

    def __len__(self) -> int:
        return len(self.labels)


class CyberEventGenerator:
    """Generates class-conditional Gaussian clusters in feature space."""

    def __init__(self, num_features: int = 32, seed: int = 42) -> None:
        self.num_features = num_features
        self.rng = np.random.default_rng(seed)

        n_fam = len(ATTACK_FAMILIES)
        # Well-separated family centers on a scaled random basis.
        self._centers = self.rng.normal(0.0, 1.0, size=(n_fam, num_features))
        self._centers *= 4.0
        # Per-family spread; benign is broad (routine noise), attacks tighter.
        self._scales = self.rng.uniform(0.4, 0.9, size=n_fam)
        self._scales[0] = 1.2  # benign

    # -- generation -------------------------------------------------------
    def sample(self, n: int, families: list[int] | None = None) -> CyberEventBatch:
        """Sample ``n`` events, optionally restricted to given family indices."""
        n_fam = len(ATTACK_FAMILIES)
        if families is None:
            families = list(range(n_fam))

        labels = self.rng.choice(families, size=n)
        feats = np.empty((n, self.num_features), dtype=np.float32)
        sev = np.empty(n, dtype=np.float32)
        etype = np.empty(n, dtype=np.int64)

        for i, lab in enumerate(labels):
            center = self._centers[lab]
            scale = self._scales[lab]
            feats[i] = center + self.rng.normal(0.0, scale, size=self.num_features)
            base = _FAMILY_SEVERITY[ATTACK_FAMILIES[lab]]
            sev[i] = float(np.clip(base + self.rng.normal(0.0, 0.05), 0.0, 1.0))
            etype[i] = self.rng.integers(0, len(EVENT_TYPES))

        return CyberEventBatch(feats, labels.astype(np.int64), sev, etype)

    def make_partial(self, features: np.ndarray, keep_ratio: float = 0.4
                     ) -> np.ndarray:
        """Mask features to simulate an incomplete / partially observed event.

        Returns a copy of ``features`` with a random subset of feature
        dimensions zeroed out. Used to test behavior-driven pattern completion.

        Note: pass *standardized* features so that a zeroed dimension equals the
        neutral feature mean; masking raw features would inject a large
        artificial offset once standardized and distort the embedding.
        """
        masked = np.asarray(features, dtype=np.float32).copy()
        n, f = masked.shape
        n_keep = max(1, int(round(f * keep_ratio)))
        for i in range(n):
            drop = self.rng.permutation(f)[n_keep:]
            masked[i, drop] = 0.0
        return masked
