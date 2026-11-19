"""MLflow store preparation.

MLflow will create the SQLite *file* but not its parent directory, which turns a
fresh checkout into an obscure "unable to open database file" error. Creating the
directory here keeps local setup a no-op for the user and has no effect on server
backends (HTTP/PostgreSQL).
"""

from __future__ import annotations

from pathlib import Path

from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)

SQLITE_PREFIX = "sqlite:///"


def ensure_tracking_store(tracking_uri: str) -> None:
    """Create the parent directory of a SQLite tracking URI when needed."""
    if not tracking_uri.startswith(SQLITE_PREFIX):
        return
    raw_path = tracking_uri[len(SQLITE_PREFIX) :]
    if raw_path in {"", ":memory:"}:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("mlflow sqlite store ready", extra={"event": "tracking_store_ready"})
