"""Run pipeline stages locally.

Examples:
    python scripts/run_pipeline.py --stage ingest
    python scripts/run_pipeline.py --stage train --no-db
    python scripts/run_pipeline.py --stage all

Stages mirror the four Airflow DAGs, and call exactly the same functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from interest_rate_mlops.config import get_settings
from interest_rate_mlops.exceptions import MLOpsError
from interest_rate_mlops.logging_utils import configure_logging
from interest_rate_mlops.pipelines import (
    run_ingestion,
    run_monitoring,
    run_promotion,
    run_training,
)

STAGES = ("ingest", "train", "promote", "monitor", "all")


def _run_stage(stage: str, settings: Any, *, fetch: bool, register: bool) -> dict[str, Any]:
    if stage == "ingest":
        return run_ingestion(fetch=fetch, settings=settings).as_dict()
    if stage == "train":
        return run_training(settings=settings, register=register).as_dict()
    if stage == "promote":
        outcome = run_promotion(settings=settings)
        return outcome.as_dict()
    if stage == "monitor":
        return run_monitoring(settings=settings).as_dict()
    raise ValueError(f"unknown stage {stage!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--fetch", action="store_true", help="Download the source first")
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow registration")
    parser.add_argument("--no-db", action="store_true", help="Skip database persistence")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.no_db:
        settings = settings.model_copy(update={"enable_db_persistence": False})
    configure_logging(args.log_level or settings.log_level, "console")

    stages = STAGES[:-1] if args.stage == "all" else (args.stage,)
    summary: dict[str, Any] = {}

    for stage in stages:
        try:
            result = _run_stage(stage, settings, fetch=args.fetch, register=not args.no_register)
        except MLOpsError as exc:
            print(f"[{stage}] failed: {exc}", file=sys.stderr)
            return 1
        summary[stage] = result
        print(f"[{stage}] ok")
        print(json.dumps(result, indent=2, default=str)[:4000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
