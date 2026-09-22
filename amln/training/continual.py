"""Training routines.

Two stages, matching the paper's design:

1. **Encoder pre-training** - a short autoencoder fit so the 64-d latent space
   captures behavioral semantics. This is done once.
2. **Continual landscape population** - events are streamed through the frozen
   encoder into the adaptive memory landscape *incrementally*, with no network
   retraining, which is what limits catastrophic forgetting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from amln.config import AMLNConfig
from amln.memory.landscape import AdaptiveMemoryLandscape, RegionType
from amln.models.encoder import BehaviorAutoencoder

if TYPE_CHECKING:
    from amln.memory.graph import CyberMemoryGraph


def pretrain_autoencoder(model: BehaviorAutoencoder, features: np.ndarray,
                         cfg: AMLNConfig) -> BehaviorAutoencoder:
    """Fit the encoder/decoder as a reconstruction autoencoder."""
    device = torch.device(cfg.train.device)
    model.to(device)
    model.train()

    x = torch.from_numpy(features.astype(np.float32))
    loader = DataLoader(TensorDataset(x), batch_size=cfg.train.batch_size,
                        shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    loss_fn = nn.MSELoss()

    mask_max = float(cfg.train.denoise_mask)
    for epoch in range(cfg.train.epochs):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            # Denoising: zero a random subset of input features per sample
            # (0 == neutral mean in standardized space); target stays clean.
            noisy = batch
            if mask_max > 0:
                keep = 1.0 - torch.rand(batch.size(0), 1, device=device) * mask_max
                mask = (torch.rand_like(batch) < keep).float()
                noisy = batch * mask
            opt.zero_grad()
            _, recon = model(noisy)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        avg = total / len(x)
        print(f"  [pretrain] epoch {epoch + 1}/{cfg.train.epochs}  recon_mse={avg:.4f}")
    return model


def populate_landscape(model: BehaviorAutoencoder,
                       landscape: AdaptiveMemoryLandscape,
                       features: np.ndarray, labels: np.ndarray,
                       severity: np.ndarray, cfg: AMLNConfig,
                       log_every: int = 0,
                       graph: "CyberMemoryGraph | None" = None,
                       metrics_log: list | None = None,
                       ) -> AdaptiveMemoryLandscape:
    """Stream events through the encoder into the landscape, one at a time.

    If ``graph`` is provided, temporal transitions are recorded in it.
    If ``metrics_log`` is provided, snapshot dicts are appended at log_every intervals.
    """
    device = torch.device(cfg.train.device)
    embeddings = _embed_all(model, features, cfg, device)

    for i in range(len(features)):
        node = landscape.observe(embeddings[i], float(severity[i]), int(labels[i]))
        # Record transition in the Cyber Memory Graph
        if graph is not None:
            node_idx = _find_node_idx(landscape, node)
            if node_idx >= 0:
                graph.record_transition(node_idx)

        if log_every and (i + 1) % log_every == 0:
            reg = landscape.regions()
            print(f"  [continual] {i + 1:>6} events | nodes={len(landscape):>4} "
                  f"| hills={reg['hill']} slopes={reg['slope']} "
                  f"| compression={landscape.compression_ratio():.1f}x")
            if metrics_log is not None:
                metrics_log.append({
                    "step": i + 1,
                    "total_nodes": len(landscape),
                    "hills": reg[RegionType.HILL],
                    "slopes": reg[RegionType.SLOPE],
                    "compression": landscape.compression_ratio(),
                    "forgetting_eff": landscape.forgetting_efficiency(),
                })
    return landscape


def _find_node_idx(landscape: AdaptiveMemoryLandscape, node) -> int:
    """Find index of a node in the landscape (by identity)."""
    for i, n in enumerate(landscape.nodes):
        if n is node:
            return i
    return -1


def _embed_all(model: BehaviorAutoencoder, features: np.ndarray,
               cfg: AMLNConfig, device: torch.device) -> np.ndarray:
    """Batch-encode all events to 64-d embeddings (numpy)."""
    model.to(device)
    x = torch.from_numpy(features.astype(np.float32))
    out = []
    with torch.no_grad():
        for start in range(0, len(x), cfg.train.batch_size):
            chunk = x[start:start + cfg.train.batch_size].to(device)
            out.append(model.embed(chunk).cpu().numpy())
    return np.concatenate(out, axis=0)
