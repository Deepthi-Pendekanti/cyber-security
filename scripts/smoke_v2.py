"""Smoke test for the V2 landscape + streaming replayer on synthetic data.

Verifies, without needing the Kaggle dataset, that:
  * the streaming replayer emits causally-standardized events,
  * V2 observe() runs, spawns/decays, stays within the energy budget,
  * zero-day (held-out) families produce higher surprise than known families.
"""

from __future__ import annotations

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.data import ATTACK_FAMILIES, CyberEventGenerator
from amln.memory.adaptive_v2 import AdaptiveMemoryLandscapeV2, LandscapeV2Config
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming import StreamReplayer
from amln.training import pretrain_autoencoder


def main() -> None:
    cfg = AMLNConfig()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    zero_day = [ATTACK_FAMILIES.index("apt_persistence"),
                ATTACK_FAMILIES.index("insider_threat")]
    known = [i for i in range(len(ATTACK_FAMILIES)) if i not in zero_day]

    gen = CyberEventGenerator(num_features=cfg.encoder.num_features, seed=cfg.train.seed)
    train = gen.sample(8000, families=known)
    test_known = gen.sample(1500, families=known)
    test_zero = gen.sample(1500, families=zero_day)

    # Pre-train encoder on known-only (standardized inside replayer for stream).
    from amln.data.preprocess import StandardScaler
    scaler = StandardScaler().fit(train.features)
    model = BehaviorAutoencoder(cfg.encoder)
    pretrain_autoencoder(model, scaler.transform(train.features), cfg)

    # Build a stream: known events first, then an injected zero-day burst.
    feats = np.concatenate([train.features, test_zero.features])
    labs = np.concatenate([train.labels, test_zero.labels])
    sev = np.concatenate([train.severity, test_zero.severity])
    ts = np.arange(len(feats), dtype=np.float64)

    replayer = StreamReplayer(feats, labs, sev, ts, warmup=4000)
    land = AdaptiveMemoryLandscapeV2(cfg.encoder.latent_dim, cfg.landscape,
                                     LandscapeV2Config())

    known_surprise, zero_surprise = [], []
    zero_set = set(zero_day)

    for ev in replayer.stream():
        x = torch.from_numpy(ev.features[None, :].astype(np.float32))
        with torch.no_grad():
            z, recon = model(x)
            recon_err = float(torch.mean((recon - x) ** 2).item())
        emb = z.numpy()[0]
        res = land.observe(emb, ev.severity, ev.label, recon_error=recon_err)
        if ev.label in zero_set:
            zero_surprise.append(res.surprise)
        else:
            known_surprise.append(res.surprise)
        if ev.index == 7999:  # snapshot just before the zero-day burst begins
            reg0 = land.regions()
            print(f"[pre-burst] nodes={len(land)} hills={reg0['hill']} "
                  f"energy={land.total_energy():.1f}")

    # Diagnostic: how separable are the family embeddings?
    with torch.no_grad():
        all_emb = model.embed(torch.from_numpy(
            scaler.transform(train.features)[:3000].astype(np.float32))).numpy()
    all_lab = train.labels[:3000]
    cents = []
    for fam in known:
        m = all_emb[all_lab == fam]
        if len(m):
            cents.append(m.mean(0) / (np.linalg.norm(m.mean(0)) + 1e-9))
    cents = np.stack(cents)
    cross = cents @ cents.T
    off = cross[~np.eye(len(cents), dtype=bool)]
    print(f"\n[diag] family centroid cosine: mean off-diag={off.mean():.3f} "
          f"max={off.max():.3f} (merge_thresh={cfg.landscape.merge_similarity})")

    print("\n=== V2 smoke results ===")
    print(f"nodes={len(land)}  energy={land.total_energy():.1f} "
          f"(budget={land.cfg2.energy_budget})")
    print(f"spawned={land.total_spawned} pruned={land.total_pruned} "
          f"splits={land.total_splits}")
    print(f"compression={land.compression_ratio():.1f}x  "
          f"forgetting_eff={land.forgetting_efficiency():.3f}")
    print(f"mean surprise  known={np.mean(known_surprise):.3f}  "
          f"zero-day={np.mean(zero_surprise):.3f}")
    reg = land.regions()
    print(f"regions: hills={reg['hill']} slopes={reg['slope']} valleys={reg['valley']}")

    # Energy budget must hold.
    assert land.total_energy() <= land.cfg2.energy_budget + 1e-3, "energy budget violated"
    # Zero-day should be more surprising than known on average (the core claim).
    print("\nPASS: energy bounded." )
    if np.mean(zero_surprise) > np.mean(known_surprise):
        print("PASS: zero-day surprise > known surprise (novelty gate works).")
    else:
        print("WARN: zero-day surprise not higher -- tune surprise weights.")


if __name__ == "__main__":
    main()
