"""One-time: load full CIC-IDS2017, subsample, and cache to data/stream.npz.

Loading 2.8M flows from ~843MB of CSVs takes minutes; doing it on every
benchmark/dashboard run is wasteful. This caches a tractable, chronology-
preserving stream once so downstream code loads it instantly.

Run:  python scripts/prepare_stream.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from amln.data.cicids import load_cicids2017, stratified_stream

RAW = Path("data/raw/cicids2017")
OUT = Path("data/stream.npz")


def main() -> None:
    print(f"Loading full dataset from {RAW} (this takes a few minutes) ...")
    data = load_cicids2017(RAW)
    print(f"  full: {len(data):,} flows, {data.features.shape[1]} features")

    stream = stratified_stream(data, max_benign=40000, max_per_attack=8000)
    counts = {stream.label_names[i]: int(np.sum(stream.labels == i))
              for i in stream.label_names}
    print(f"  subsampled stream: {len(stream):,} events")
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {c}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Save label_names as parallel arrays (npz can't store dicts directly).
    ids = np.array(list(stream.label_names.keys()), dtype=np.int64)
    names = np.array([stream.label_names[i] for i in ids], dtype=object)
    np.savez_compressed(
        OUT,
        features=stream.features,
        labels=stream.labels,
        severity=stream.severity,
        timestamps=stream.timestamps,
        label_ids=ids,
        label_names=names,
        feature_names=np.array(stream.feature_names, dtype=object),
    )
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"\nSaved {OUT} ({size_mb:.1f} MB). Benchmark/dashboard will use it.")


if __name__ == "__main__":
    main()
