"""Generate the EDA report from the ingested monthly panel.

Usage:
    python scripts/run_eda.py [--output-dir reports/eda]
"""

from __future__ import annotations

import argparse
import json
import sys

from interest_rate_mlops.config import get_settings, load_data_config
from interest_rate_mlops.eda import build_eda_report
from interest_rate_mlops.exceptions import MLOpsError
from interest_rate_mlops.logging_utils import configure_logging
from interest_rate_mlops.pipelines.ingestion_pipeline import ensure_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="Defaults to reports/eda")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, "console")
    data_cfg = load_data_config()

    try:
        panel, metadata = ensure_panel(data_cfg)
    except MLOpsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or (settings.reports_path / "eda")
    payload = build_eda_report(
        panel,
        output_dir,
        target=data_cfg.target.slug,
        unit=data_cfg.target.unit,
        extra={
            "raw_rows": metadata.get("rows_in"),
            "intra_period_rows": metadata.get("intra_period_rows_dropped"),
            "dataset_version": metadata.get("dataset_version"),
        },
    )
    print(json.dumps(payload["window"], indent=2))
    print(f"report: {output_dir / 'eda_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
