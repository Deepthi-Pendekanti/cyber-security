"""CIC-IDS2017 loader + stream ordering for AMLN.

The CIC-IDS2017 "MachineLearningCVE" bundle ships one CSV per capture session
(Monday benign, then Tuesday..Friday with attacks injected morning/afternoon).
Each row is a bidirectional network flow with ~78 numeric features and a
``Label`` column (``BENIGN`` or an attack name).

This loader:
  * reads every daily CSV (optionally capped),
  * normalizes the messy column names (CIC columns have stray spaces),
  * coerces features to float, replacing NaN/Inf (the dataset has many),
  * concatenates files in capture order so a replay follows the real
    Monday->Friday timeline (the "real-time" story),
  * maps attack names to severity using a curated table,
  * returns AMLN's (features, labels, severity, timestamps, label_names).

Nothing is standardized here; fit the scaler on the benign/training prefix in
the caller so the stream is treated causally (no peeking at future flows).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Capture-day ordering so a replay matches the real timeline. Matching is by
# case-insensitive substring against the CSV filename.
_DAY_ORDER = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday-morning",       # web attacks / infiltration split across files
    "thursday-afternoon",
    "thursday",
    "friday-morning",
    "friday-afternoon",
    "friday",
]

# Severity per CIC-IDS2017 attack label (0..1). Higher = more damaging.
_LABEL_SEVERITY = {
    "BENIGN": 0.05,
    "FTP-Patator": 0.55,
    "SSH-Patator": 0.55,
    "DoS Hulk": 0.80,
    "DoS GoldenEye": 0.78,
    "DoS slowloris": 0.72,
    "DoS Slowhttptest": 0.72,
    "Heartbleed": 0.95,
    "Web Attack \x96 Brute Force": 0.70,
    "Web Attack \x96 XSS": 0.68,
    "Web Attack \x96 Sql Injection": 0.85,
    "Web Attack - Brute Force": 0.70,
    "Web Attack - XSS": 0.68,
    "Web Attack - Sql Injection": 0.85,
    "Infiltration": 0.90,
    "Bot": 0.82,
    "PortScan": 0.60,
    "DDoS": 0.88,
}


@dataclass
class StreamData:
    """Chronologically ordered events ready for streaming replay."""

    features: np.ndarray      # (N, F) float32, NOT yet standardized
    labels: np.ndarray        # (N,) int family ids (0 == BENIGN)
    severity: np.ndarray      # (N,) float in [0, 1]
    timestamps: np.ndarray    # (N,) float monotonic surrogate (capture order)
    label_names: dict         # id -> string name
    feature_names: list       # column names kept as features

    def __len__(self) -> int:
        return len(self.labels)


def _day_rank(filename: str) -> int:
    f = filename.lower()
    for rank, key in enumerate(_DAY_ORDER):
        if key in f:
            return rank
    return len(_DAY_ORDER)  # unknown files sort last


def _normalize_cols(cols: list[str]) -> list[str]:
    return [c.strip() for c in cols]


def load_cicids2017(root: str | Path, max_rows_per_file: int | None = None
                    ) -> StreamData:
    """Load and chronologically order the CIC-IDS2017 flow CSVs.

    Args:
        root: directory containing the daily CSV files (searched recursively).
        max_rows_per_file: optional cap per file for quick experiments.
    """
    import pandas as pd

    root = Path(root)
    csv_files = sorted(root.rglob("*.csv"), key=lambda p: (_day_rank(p.name), p.name))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files under {root}. Run scripts/download_cicids2017.py first.")

    frames = []
    for order, path in enumerate(csv_files):
        df = pd.read_csv(path, nrows=max_rows_per_file, low_memory=False)
        df.columns = _normalize_cols(list(df.columns))
        if "Label" not in df.columns:
            # Some mirrors name it differently; try a case-insensitive match.
            label_col = next((c for c in df.columns if c.lower() == "label"), None)
            if label_col is None:
                continue
            df = df.rename(columns={label_col: "Label"})
        df["__file_order"] = order
        df["__row_order"] = np.arange(len(df))
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Chronological surrogate: files in capture order, rows within file in order.
    data = data.sort_values(["__file_order", "__row_order"]).reset_index(drop=True)
    timestamps = np.arange(len(data), dtype=np.float64)

    raw_labels = data["Label"].astype(str).str.strip().to_numpy()

    # Feature columns = everything numeric except label/order helpers.
    drop = {"Label", "__file_order", "__row_order"}
    feature_cols = [c for c in data.columns if c not in drop]
    feats = data[feature_cols].apply(_to_numeric_series).to_numpy(dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # Label ids with BENIGN pinned to 0.
    uniq = sorted(set(raw_labels), key=lambda s: (s.upper() != "BENIGN", s))
    name_to_id = {name: i for i, name in enumerate(uniq)}
    labels = np.asarray([name_to_id[l] for l in raw_labels], dtype=np.int64)

    severity = np.asarray(
        [_LABEL_SEVERITY.get(l, 0.5 if l.upper() != "BENIGN" else 0.05)
         for l in raw_labels], dtype=np.float32)

    return StreamData(
        features=feats,
        labels=labels,
        severity=severity,
        timestamps=timestamps,
        label_names={i: n for n, i in name_to_id.items()},
        feature_names=feature_cols,
    )


def stratified_stream(data: StreamData, max_benign: int = 40000,
                      max_per_attack: int = 8000, seed: int = 42) -> StreamData:
    """Subsample a tractable, chronology-preserving stream from the full dataset.

    CIC-IDS2017 has 2.8M flows (80% benign) -- too many to encode quickly, and
    so imbalanced that rare attacks (Bot, Heartbleed, Web/Infiltration) vanish
    under a naive cap. This keeps:
      * every event of any attack family up to ``max_per_attack`` (so rare
        families survive), and
      * a uniform chronological sample of benign flows up to ``max_benign``,
    then re-sorts by the original timestamp so the Monday->Friday attack
    timeline is preserved for streaming replay.
    """
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(data), dtype=bool)

    for label_id in data.label_names:
        idx = np.where(data.labels == label_id)[0]
        cap = max_benign if label_id == 0 else max_per_attack
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        keep[idx] = True

    sel = np.where(keep)[0]
    sel = sel[np.argsort(data.timestamps[sel], kind="stable")]
    return StreamData(
        features=data.features[sel],
        labels=data.labels[sel],
        severity=data.severity[sel],
        timestamps=data.timestamps[sel],
        label_names=data.label_names,
        feature_names=data.feature_names,
    )


def _to_numeric_series(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")
