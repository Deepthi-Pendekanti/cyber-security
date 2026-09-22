"""Transformer encoder that maps a cyber event to a 64-d behavioral embedding.

The event's structured features are treated as a short token sequence (one
token per feature), passed through a standard Transformer encoder, mean-pooled,
and projected to the latent space. A lightweight decoder reconstructs the full
feature vector; training the pair as an autoencoder yields embeddings that
capture *behavioral semantics* rather than raw values, and the decoder later
powers behavior-driven pattern completion.
"""

from __future__ import annotations

import torch
from torch import nn

from amln.config import EncoderConfig


class TransformerEncoder(nn.Module):
    """Feature-token Transformer -> L2-normalized 64-d embedding."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Per-feature scalar -> d_model token embedding, plus learned position.
        self.value_proj = nn.Linear(1, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(cfg.num_features, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.to_latent = nn.Linear(cfg.d_model, cfg.latent_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # x: (B, num_features) -> tokens (B, num_features, d_model)
        tokens = self.value_proj(x.unsqueeze(-1)) + self.pos_emb.unsqueeze(0)
        encoded = self.encoder(tokens)          # (B, F, d_model)
        pooled = encoded.mean(dim=1)            # (B, d_model)
        latent = self.to_latent(pooled)         # (B, latent_dim)
        if normalize:
            latent = torch.nn.functional.normalize(latent, dim=-1)
        return latent


class _Decoder(nn.Module):
    """Latent -> reconstructed feature vector."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, cfg.num_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BehaviorAutoencoder(nn.Module):
    """Encoder + decoder trained jointly to reconstruct event features."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(cfg)
        self.decoder = _Decoder(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)

    @torch.no_grad()
    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.decoder(z)
