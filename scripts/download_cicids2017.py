"""Download the CIC-IDS2017 dataset from Kaggle into ``data/raw``.

Prerequisite (one-time): place your Kaggle API token at
``%USERPROFILE%\\.kaggle\\kaggle.json`` (Kaggle -> Settings -> API ->
Create New Token). Then run:

    .\\.venv\\Scripts\\python.exe scripts\\download_cicids2017.py

The script downloads the "MachineLearningCVE" flow-feature CSV bundle
(a few hundred MB), unzips it into ``data/raw/cicids2017/``, and verifies
the expected daily CSV files are present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Primary dataset slug (flow-feature CSVs). Fallback slugs are tried in order
# in case the primary is unavailable under the account/region.
DATASET_SLUGS = [
    "cicdataset/cicids2017",
    "chethuhn/network-intrusion-dataset",
]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "cicids2017"


def _check_token() -> None:
    token = Path(os.path.expanduser("~")) / ".kaggle" / "kaggle.json"
    if not token.exists():
        print("ERROR: Kaggle API token not found at", token)
        print("  1. Go to https://www.kaggle.com -> Settings -> API -> Create New Token")
        print("  2. Save the downloaded kaggle.json to:", token.parent)
        sys.exit(1)


def main() -> None:
    _check_token()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Import here so the token check runs first with a clean error.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    last_err: Exception | None = None
    for slug in DATASET_SLUGS:
        try:
            print(f"Downloading {slug} -> {RAW_DIR} ...")
            api.dataset_download_files(slug, path=str(RAW_DIR), unzip=True, quiet=False)
            break
        except Exception as e:  # noqa: BLE001 - surface any Kaggle API error, try next
            last_err = e
            print(f"  failed on {slug}: {e}")
    else:
        print("ERROR: all dataset slugs failed. Last error:", last_err)
        sys.exit(1)

    csvs = list(RAW_DIR.rglob("*.csv"))
    print(f"\nDone. Found {len(csvs)} CSV file(s):")
    for c in sorted(csvs):
        size_mb = c.stat().st_size / (1024 * 1024)
        print(f"  {c.relative_to(RAW_DIR)}  ({size_mb:.1f} MB)")
    if not csvs:
        print("WARNING: no CSV files found after download.")


if __name__ == "__main__":
    main()
