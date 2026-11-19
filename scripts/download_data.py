"""Download raw data.

Kaggle (default, the project's initial source):
    python scripts/download_data.py --source kaggle

Official refresh from FRED (no API key needed):
    python scripts/download_data.py --source fred --series FEDFUNDS
"""

from __future__ import annotations

import argparse
import sys

from interest_rate_mlops.config import get_settings, load_data_config
from interest_rate_mlops.ingestion import fetch_fred_series, fetch_kaggle_dataset
from interest_rate_mlops.logging_utils import configure_logging


def main() -> int:
    data_cfg = load_data_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["kaggle", "fred"], default=data_cfg.source)
    parser.add_argument("--series", action="append", default=None, help="FRED series id(s)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, "console")

    if args.source == "kaggle":
        if not data_cfg.kaggle_dataset:
            print("error: dataset.kaggle_dataset is not configured", file=sys.stderr)
            return 2
        path = fetch_kaggle_dataset(data_cfg.kaggle_dataset, data_cfg.raw_directory, args.force)
        print(f"downloaded: {path}")
        print(
            "note: the configured raw file is "
            f"{data_cfg.raw_path.name}; update configs/data.yaml if the archive "
            "contains a different filename."
        )
        return 0

    series = args.series or data_cfg.fred_series
    if not series:
        print("error: pass --series <ID> for a FRED download", file=sys.stderr)
        return 2
    for series_id in series:
        path = fetch_fred_series(series_id, data_cfg.raw_directory, args.force)
        print(f"downloaded: {path}")
    print("point configs/data.yaml at the FRED CSV to ingest it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
