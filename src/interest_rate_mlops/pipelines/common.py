"""Helpers shared by the pipeline entry points."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from interest_rate_mlops.config import Settings
from interest_rate_mlops.database.session import create_engine_from_url, create_session_factory
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.storage import read_json, write_json

logger = get_logger(__name__)


@contextmanager
def optional_session(settings: Settings, *, required: bool = False) -> Iterator[Session | None]:
    """Yield a session when the database is reachable, otherwise ``None``.

    Training and monitoring must still produce artifacts on a machine where
    PostgreSQL is not running (a common local-development state). Unavailability
    is logged as a warning and surfaced in the pipeline outcome - never hidden.
    """
    if not settings.enable_db_persistence:
        yield None
        return

    engine = None
    try:
        engine = create_engine_from_url(
            settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
        )
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        if required:
            raise
        logger.warning(
            "database unavailable - continuing without persistence (%s)",
            str(exc).splitlines()[0][:200],
            extra={"event": "database_unavailable", "status_code": 503},
        )
        if engine is not None:
            engine.dispose()
        yield None
        return

    factory = create_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


class Timer:
    """Context manager measuring wall-clock duration in seconds."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.elapsed: float | None = None

    def __enter__(self) -> Timer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = time.perf_counter() - self.started

    @property
    def seconds(self) -> float:
        return float(self.elapsed or 0.0)


def dvc_add(paths: list[str | Path], project_root: Path) -> dict[str, Any]:
    """Best-effort ``dvc add`` for pipeline outputs.

    DVC is a repository-level tool: it writes ``.dvc`` pointer files next to the
    data. The pipeline records the outcome instead of failing when DVC is absent
    or the working tree is not a DVC repo (e.g. inside a read-only container).

    Paths that ``dvc.yaml`` already declares as stage outputs are skipped: those
    are owned by the DVC pipeline, and adding them again would conflict.
    """
    targets = [str(Path(path)) for path in paths if Path(path).exists()]
    if not targets:
        return {"ok": False, "reason": "no existing paths to track", "tracked": []}

    owned = _dvc_pipeline_outputs(project_root)
    if owned:
        kept = [target for target in targets if _relative(target, project_root) not in owned]
        skipped = sorted(set(targets) - set(kept))
        if not kept:
            return {
                "ok": True,
                "reason": "outputs are owned by the dvc.yaml pipeline",
                "tracked": [],
                "skipped": [_relative(target, project_root) for target in skipped],
            }
        targets = kept

    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "dvc", "add", *targets],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        logger.warning("DVC tracking skipped: %s", exc, extra={"event": "dvc_skipped"})
        return {"ok": False, "reason": str(exc), "tracked": []}

    ok = result.returncode == 0
    if not ok:
        logger.warning(
            "DVC add returned %s: %s",
            result.returncode,
            (result.stderr or result.stdout).strip()[:500],
            extra={"event": "dvc_failed"},
        )
    return {
        "ok": ok,
        "reason": (result.stdout or result.stderr).strip()[:1000],
        "tracked": targets,
    }


def _relative(path: str | Path, project_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(project_root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _dvc_pipeline_outputs(project_root: Path) -> set[str]:
    """Declared ``outs`` of dvc.yaml, as posix paths relative to the repo root."""
    pipeline_file = project_root / "dvc.yaml"
    if not pipeline_file.exists():
        return set()
    try:
        document = yaml.safe_load(pipeline_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()

    declared: set[str] = set()
    for stage in (document.get("stages") or {}).values():
        for entry in stage.get("outs") or []:
            path = entry if isinstance(entry, str) else next(iter(entry), None)
            if isinstance(path, str) and path:
                declared.add(Path(path).as_posix())
    return declared


# ---------------------------------------------------------------------------
# Pipeline state file (scraped by the Prometheus file collector)
# ---------------------------------------------------------------------------
def state_file(settings: Settings) -> Path:
    return Path(settings.reports_path) / "state" / "pipeline_status.json"


def record_pipeline_state(
    settings: Settings,
    *,
    pipeline: str,
    status: str,
    duration_seconds: float,
    details: dict[str, Any] | None = None,
) -> Path:
    """Append this run to the state document the metrics collector reads."""
    path = state_file(settings)
    document = read_json(path, default={}) or {}
    if not isinstance(document, dict):
        document = {}
    document[pipeline] = {
        "status": status,
        "duration_seconds": round(float(duration_seconds), 4),
        "timestamp": utc_now().isoformat(),
        "details": details or {},
    }
    return write_json(path, document)
