"""Structured logging.

Application logs are emitted as single-line JSON by default so they can be
shipped and queried by a log aggregator; a readable console format is available
for local development via ``LOG_FORMAT=console``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False

# Attributes promoted to top-level JSON fields when supplied via ``extra=``.
EXTRA_FIELDS = (
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "model_name",
    "model_version",
    "run_id",
    "pipeline",
    "dataset_version",
    "task",
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        # Any additional ``extra=`` keys are preserved too.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Readable single-line format for interactive development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(level: str = "INFO", fmt: str = "json", force: bool = False) -> None:
    """Configure the root logger once per process (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party libraries are noisy at INFO.
    for noisy in ("urllib3", "botocore", "alembic.runtime.migration", "git.cmd"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
