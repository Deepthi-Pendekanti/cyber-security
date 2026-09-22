"""Loaders for real public cybersecurity datasets.

These are thin, dependency-light CSV loaders that map a real dataset onto the
same (features, labels, severity) contract the synthetic generator uses, so the
full AMLN pipeline runs unchanged. Column handling is intentionally generic;
adjust ``label_column`` / ``drop_columns`` per dataset.

Supported out of the box (CSV form): CIC-IDS2017/2018, UNSW-NB15, CIC-MalMem.
EMBER ships as JSONL/vectorized features and needs its own loader.
"""

from __future__ import annotations

import csv

import numpy as np

from amln.data.preprocess import StandardScaler


def load_csv_dataset(path: str, label_column: str,
                     drop_columns: tuple[str, ...] = (),
                     benign_label: str = "BENIGN",
                     max_rows: int | None = None) -> dict:
    """Load a flow/telemetry CSV into AMLN's arrays.

    Returns a dict with keys: ``features`` (standardized float32),
    ``labels`` (int family ids), ``severity`` (0/1 attack flag as a proxy),
    ``label_names`` (id -> string), and ``scaler``.
    """
    rows: list[list[float]] = []
    raw_labels: list[str] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None, "CSV has no header row"
        feature_cols = [c for c in reader.fieldnames
                        if c != label_column and c not in drop_columns]
        for n, record in enumerate(reader):
            if max_rows is not None and n >= max_rows:
                break
            try:
                vals = [_to_float(record[c]) for c in feature_cols]
            except (KeyError, ValueError):
                continue
            rows.append(vals)
            raw_labels.append(str(record[label_column]).strip())

    features = np.asarray(rows, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Map string labels -> ids, keeping benign at index 0.
    uniq = sorted(set(raw_labels), key=lambda s: (s.upper() != benign_label.upper(), s))
    name_to_id = {name: i for i, name in enumerate(uniq)}
    labels = np.asarray([name_to_id[l] for l in raw_labels], dtype=np.int64)

    benign_id = name_to_id.get(benign_label, name_to_id.get(benign_label.upper(), 0))
    severity = (labels != benign_id).astype(np.float32)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return {
        "features": features,
        "labels": labels,
        "severity": severity,
        "label_names": {i: n for n, i in name_to_id.items()},
        "scaler": scaler,
    }


def _to_float(value: str) -> float:
    v = value.strip()
    if v == "" or v.lower() in {"nan", "inf", "-inf"}:
        return 0.0
    return float(v)
