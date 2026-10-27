"""Quick check: does RobustScaler fix the encoder MSE on real CIC-IDS2017?

Loads the cached stream, builds a replayer with the chosen scaler, and
pre-trains the autoencoder, printing final MSE. Compares standard vs robust.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming import StreamReplayer
from amln.training import pretrain_autoencoder


def run(scaler: str, epochs: int) -> float:
    cfg = AMLNConfig()
    cfg.train.epochs = epochs
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    z = np.load(Path("data/stream.npz"), allow_pickle=True)
    feats, labels, sev, ts = (z["features"], z["labels"], z["severity"], z["timestamps"])
    cfg.encoder.num_features = feats.shape[1]

    warmup = min(5000, len(feats) // 3)
    rep = StreamReplayer(feats, labels, sev, ts, warmup=warmup, scaler=scaler)
    x = rep.standardized_block(0, warmup)
    print(f"[{scaler}] standardized stats: min={x.min():.2f} max={x.max():.2f} "
          f"mean={x.mean():.3f} std={x.std():.3f}")
    model = BehaviorAutoencoder(cfg.encoder)
    pretrain_autoencoder(model, x, cfg)
    # Final MSE on a held-out slice of the stream.
    val = rep.standardized_block(warmup, 5000)
    with torch.no_grad():
        _, recon = model(torch.from_numpy(val))
        mse = float(torch.mean((recon - torch.from_numpy(val)) ** 2).item())
    print(f"[{scaler}] held-out MSE = {mse:.4f}")
    return mse


if __name__ == "__main__":
    scaler = sys.argv[1] if len(sys.argv) > 1 else "robust"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    run(scaler, epochs)
