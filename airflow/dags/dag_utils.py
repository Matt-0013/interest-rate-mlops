"""Helpers shared by the DAGs.

Schedules come from environment variables so the same image can run the DAGs
manually for the static Kaggle dataset, or on a timetable once a live data source
is configured. An empty value means "manual trigger only", which is the default.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "depends_on_past": False,
    "email_on_failure": False,
}

START_DATE = datetime(2024, 1, 1)


def schedule(env_var: str) -> str | None:
    """Return the cron/preset schedule for a DAG, or ``None`` for manual only."""
    value = os.environ.get(env_var, "").strip()
    return value or None


def jsonable(value: Any) -> Any:
    """Make a pipeline outcome safe to pass through XCom."""
    return json.loads(json.dumps(value, default=str))


def fetch_enabled() -> bool:
    return os.environ.get("INGESTION_FETCH", "false").strip().lower() in {"1", "true", "yes"}


def auto_retrain_enabled() -> bool:
    return os.environ.get("AUTO_RETRAIN", "false").strip().lower() in {"1", "true", "yes"}
