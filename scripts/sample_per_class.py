"""Print one real example flow per class from CIC-IDS2017.

Shows a compact set of human-readable features for one representative row of
each label, so you can see exactly what the model receives as input and what
the ground-truth class is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/cicids2017")

# A readable subset of the 78 flow features (names after stripping spaces).
SHOW = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Max",
    "Average Packet Size",
]


def main() -> None:
    csvs = sorted(RAW.rglob("*.csv"))
    seen: dict[str, dict] = {}

    for path in csvs:
        # Read in chunks so we don't load everything into memory.
        for chunk in pd.read_csv(path, chunksize=200000, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if "Label" not in chunk.columns:
                continue
            for label, grp in chunk.groupby("Label"):
                label = str(label).strip()
                if label in seen:
                    continue
                row = grp.iloc[0]
                rec = {}
                for col in SHOW:
                    if col in row:
                        rec[col] = row[col]
                seen[label] = rec
        # Stop early if we've covered everything common.
    # Print.
    print(f"Found {len(seen)} classes. One example flow each:\n")
    for label in sorted(seen):
        print("=" * 70)
        print(f"CLASS (ground truth label): {label}")
        print("-" * 70)
        for k, v in seen[label].items():
            print(f"  {k:<28}: {v}")
        print()


if __name__ == "__main__":
    main()
