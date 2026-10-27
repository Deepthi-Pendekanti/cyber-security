"""Run the AMLN memory-management benchmark.

Uses CIC-IDS2017 if it has been downloaded to data/raw/cicids2017 (see
scripts/download_cicids2017.py); otherwise falls back to the synthetic
generator so the benchmark is runnable without the dataset.

Outputs:
  * a comparison table printed to stdout,
  * benchmark_results.csv,
  * benchmark_comparison.png (accuracy vs memory footprint).

Run:  python run_benchmark.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.benchmark.harness import BenchmarkHarness
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming import StreamReplayer
from amln.training import pretrain_autoencoder

CICIDS_DIR = Path("data/raw/cicids2017")
STREAM_NPZ = Path("data/stream.npz")


def _load_stream(cfg: AMLNConfig):
    """Return (features, labels, severity, timestamps, label_names, zero_day_ids)."""
    if STREAM_NPZ.exists():
        print(f"Loading cached CIC-IDS2017 stream from {STREAM_NPZ} ...")
        z = np.load(STREAM_NPZ, allow_pickle=True)
        feats = z["features"]
        labels = z["labels"]
        severity = z["severity"]
        ts = z["timestamps"]
        names = {int(i): str(n) for i, n in zip(z["label_ids"], z["label_names"])}
        cfg.encoder.num_features = feats.shape[1]
        attack_ids = [i for i in names if i != 0]
        counts = {i: int(np.sum(labels == i)) for i in attack_ids}
        zero_day = set(sorted(counts, key=counts.get)[:2]) if len(counts) >= 2 else set()
        print(f"  events={len(feats):,} families={len(names)} "
              f"zero-day={[names[i] for i in zero_day]}")
        return feats, labels, severity, ts, names, zero_day

    if CICIDS_DIR.exists() and list(CICIDS_DIR.rglob("*.csv")):
        print("No cached stream found. Run scripts/prepare_stream.py once first "
              "for a fast start. Loading full dataset now (slow) ...")
        from amln.data.cicids import load_cicids2017, stratified_stream
        data = load_cicids2017(CICIDS_DIR)
        data = stratified_stream(data, max_benign=40000, max_per_attack=8000,
                                 seed=cfg.train.seed)
        cfg.encoder.num_features = data.features.shape[1]
        attack_ids = [i for i in data.label_names if i != 0]
        counts = {i: int(np.sum(data.labels == i)) for i in attack_ids}
        zero_day = set(sorted(counts, key=counts.get)[:2]) if len(counts) >= 2 else set()
        print(f"  events={len(data):,} families={len(data.label_names)} "
              f"zero-day={[data.label_names[i] for i in zero_day]}")
        return (data.features, data.labels, data.severity, data.timestamps,
                data.label_names, zero_day)

    print("CIC-IDS2017 not found -> using synthetic generator (fallback).")
    from amln.data import ATTACK_FAMILIES, CyberEventGenerator
    gen = CyberEventGenerator(num_features=cfg.encoder.num_features, seed=cfg.train.seed)
    zero_day = {ATTACK_FAMILIES.index("apt_persistence"),
                ATTACK_FAMILIES.index("insider_threat")}
    batch = gen.sample(15000)
    names = {i: n for i, n in enumerate(ATTACK_FAMILIES)}
    ts = np.arange(len(batch), dtype=np.float64)
    return (batch.features, batch.labels, batch.severity, ts, names, zero_day)


def main() -> None:
    cfg = AMLNConfig()
    if torch.cuda.is_available():
        cfg.train.device = "cuda"
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    feats, labels, severity, ts, names, zero_day = _load_stream(cfg)

    warmup = min(5000, len(feats) // 3)
    replayer = StreamReplayer(feats, labels, severity, ts, warmup=warmup)

    # Pre-train the encoder on the standardized warmup prefix only (causal).
    print("\nPre-training behavioral autoencoder on warmup prefix ...")
    train_x = replayer.standardized_block(0, warmup)
    model = BehaviorAutoencoder(cfg.encoder)
    pretrain_autoencoder(model, train_x, cfg)

    print("\nRunning benchmark (prequential test-then-train) ...")
    harness = BenchmarkHarness(model, replayer, cfg, cfg.encoder.latent_dim,
                               benign_label=0, zero_day_labels=zero_day)
    results = harness.run_all(window=warmup, reservoir_k=warmup, kmeans_k=32)

    # -- table --------------------------------------------------------------
    rows = [r.as_row() for r in results]
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("\n" + "=" * 100)
    print("MEMORY-MANAGEMENT BENCHMARK")
    print("=" * 100)
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("-" * 100)
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    # -- CSV ----------------------------------------------------------------
    out_csv = Path("benchmark_results.csv")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv}")

    # -- plot ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        meths = [r.name for r in results]
        f1 = [r.macro_f1 for r in results]
        kb = [r.footprint_kb for r in results]
        colors = ["#d62728" if "AMLN" in m else "#1f77b4" for m in meths]

        ax1.bar(range(len(meths)), f1, color=colors)
        ax1.set_xticks(range(len(meths)))
        ax1.set_xticklabels(meths, rotation=30, ha="right")
        ax1.set_ylabel("Macro-F1 (prequential)")
        ax1.set_title("Detection quality")
        ax1.set_ylim(0, 1)

        ax2.scatter(kb, f1, c=colors, s=120)
        for m, x, y in zip(meths, kb, f1):
            ax2.annotate(m, (x, y), fontsize=8, xytext=(5, 5),
                         textcoords="offset points")
        ax2.set_xlabel("Memory footprint (KB, peak)")
        ax2.set_ylabel("Macro-F1")
        ax2.set_title("Accuracy vs memory  (upper-left is best)")
        if max(kb) / max(1e-6, min(kb)) > 50:
            ax2.set_xscale("log")

        fig.tight_layout()
        fig.savefig("benchmark_comparison.png", dpi=130)
        print("Saved benchmark_comparison.png")
    except ImportError:
        print("(matplotlib missing -> skipped plot)")

    # -- highlight AMLN's position -----------------------------------------
    amln = next(r for r in results if "AMLN" in r.name)
    store = next(r for r in results if r.name == "StoreAll")
    print("\n" + "=" * 100)
    print("HEADLINE")
    print("=" * 100)
    print(f"  AMLN-V2 macro-F1 = {amln.macro_f1:.4f}  vs  StoreAll = {store.macro_f1:.4f}")
    print(f"  AMLN-V2 footprint = {amln.footprint_kb:.2f} KB  vs  StoreAll = "
          f"{store.footprint_kb:.2f} KB  "
          f"({store.footprint_kb / max(1e-6, amln.footprint_kb):.0f}x smaller)")
    print(f"  AMLN-V2 zero-day recall = {amln.zero_day_recall:.4f}  vs  StoreAll = "
          f"{store.zero_day_recall:.4f}")
    print(f"  AMLN-V2 extra: {amln.extra}")


if __name__ == "__main__":
    main()
