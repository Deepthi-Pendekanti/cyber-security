"""Headless smoke test for the LiveEngine (what the dashboard drives)."""

from __future__ import annotations

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.data import ATTACK_FAMILIES, CyberEventGenerator
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming import StreamReplayer
from amln.streaming.engine import LiveEngine
from amln.training import pretrain_autoencoder


def main() -> None:
    cfg = AMLNConfig()
    cfg.train.epochs = 3
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    gen = CyberEventGenerator(num_features=cfg.encoder.num_features, seed=cfg.train.seed)
    batch = gen.sample(3000)
    names = {i: n for i, n in enumerate(ATTACK_FAMILIES)}
    ts = np.arange(len(batch), dtype=np.float64)

    replayer = StreamReplayer(batch.features, batch.labels, batch.severity, ts,
                              warmup=1500)
    model = BehaviorAutoencoder(cfg.encoder)
    pretrain_autoencoder(model, replayer.standardized_block(0, 1500), cfg)

    engine = LiveEngine(model, replayer, cfg, names)
    last = None
    steps = 0
    while True:
        snap = engine.step()
        if snap is None:
            break
        last = snap
        steps += 1

    assert last is not None
    print(f"\nPASS engine: steps={steps} processed={engine.processed} "
          f"alerts={engine.alerts}")
    print(f"  final: nodes={last.num_nodes} hills={last.hills} "
          f"energy={last.total_energy}/{last.energy_budget} "
          f"rolling_acc={last.rolling_accuracy:.3f} compression={last.compression}x")
    assert len(last.node_heights) == last.num_nodes
    print("PASS: snapshot shape consistent.")


if __name__ == "__main__":
    main()
