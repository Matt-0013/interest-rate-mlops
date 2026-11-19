"""Data ingestion pipeline (Airflow DAG 1, ``make ingest``).

Load -> profile -> validate -> build monthly panel -> validate panel -> persist
-> DVC-track -> record metadata. A validation failure is persisted before the
exception propagates, so a broken dataset always leaves an audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from interest_rate_mlops.config import (
    PROJECT_ROOT,
    DatasetConfig,
    Settings,
    get_settings,
    load_data_config,
)
from interest_rate_mlops.database.repositories import record_ingestion, record_pipeline_execution
from interest_rate_mlops.exceptions import IngestionError
from interest_rate_mlops.ingestion import (
    discover_raw_files,
    fetch_fred_series,
    fetch_kaggle_dataset,
    load_raw_frame,
    profile_raw_dataset,
)
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.pipelines.common import (
    Timer,
    dvc_add,
    optional_session,
    record_pipeline_state,
    utc_stamp,
)
from interest_rate_mlops.preprocessing import build_monthly_panel, load_panel, save_panel
from interest_rate_mlops.storage import write_json
from interest_rate_mlops.validation import CheckResult, validate_panel, validate_raw_dataset
from interest_rate_mlops.versioning import build_version_record

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionOutcome:
    """Result of one ingestion run."""

    raw_path: str
    panel_path: str
    dataset_version: str
    rows_in: int
    rows_out: int
    panel_start: str
    panel_end: str
    validation_passed: bool
    report_paths: dict[str, str] = field(default_factory=dict)
    dvc: dict[str, Any] = field(default_factory=dict)
    db_persisted: bool = False
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_path": self.raw_path,
            "panel_path": self.panel_path,
            "dataset_version": self.dataset_version,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "panel_start": self.panel_start,
            "panel_end": self.panel_end,
            "validation_passed": self.validation_passed,
            "report_paths": dict(self.report_paths),
            "dvc": dict(self.dvc),
            "db_persisted": self.db_persisted,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def _resolve_raw_path(cfg: DatasetConfig, fetch: bool) -> Path:
    """Fetch (when asked) and return the raw CSV to ingest."""
    if fetch:
        if cfg.source == "kaggle":
            if cfg.kaggle_dataset is None:  # pragma: no cover - config validator
                raise IngestionError("source is kaggle but dataset.kaggle_dataset is unset")
            fetch_kaggle_dataset(cfg.kaggle_dataset, cfg.raw_directory, force=True)
        elif cfg.source == "fred":
            for series in cfg.fred_series:
                fetch_fred_series(series, cfg.raw_directory, force=True)
        else:
            logger.info(
                "source is local_file; nothing to download", extra={"event": "local_source"}
            )

    if cfg.raw_path.exists():
        return cfg.raw_path

    candidates = discover_raw_files(cfg.raw_directory)
    if len(candidates) > 1:
        logger.warning(
            "multiple raw CSV files found; using the first one",
            extra={"event": "multiple_raw_files"},
        )
    logger.info("configured raw file missing; using %s", candidates[0])
    return candidates[0]


def run_ingestion(
    *,
    fetch: bool = False,
    settings: Settings | None = None,
    data_cfg: DatasetConfig | None = None,
) -> IngestionOutcome:
    """Run the full ingestion + validation pipeline.

    Args:
        fetch: download the configured source first (Kaggle/FRED).
        settings: environment settings (defaults to the process singleton).
        data_cfg: dataset configuration (defaults to ``configs/data.yaml``).

    Raises:
        DataValidationError: when the raw data or panel fails a hard gate.
        IngestionError: when the dataset cannot be located or parsed.
    """
    settings = settings or get_settings()
    cfg = data_cfg or load_data_config()

    stamp = utc_stamp()
    report_dir = Path(settings.reports_path) / "data_quality" / stamp
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_path = _resolve_raw_path(cfg, fetch)
    raw = load_raw_frame(raw_path)
    profile = profile_raw_dataset(raw, cfg)
    write_json(report_dir / "raw_profile.json", profile.as_dict())

    validation_report, outliers = validate_raw_dataset(raw, cfg)
    versions = build_version_record(raw_path)
    write_json(
        report_dir / "validation_report.json",
        {
            **validation_report.as_dict(),
            "outliers": [
                {
                    "column": item.column,
                    "n_outliers": item.n_outliers,
                    "lower_bound": item.lower_bound,
                    "upper_bound": item.upper_bound,
                    "samples": item.samples,
                }
                for item in outliers
            ],
            "versions": versions.as_dict(),
        },
    )

    rows_out = 0
    panel_start = ""
    panel_end = ""
    panel_path = cfg.panel_path
    dvc_result: dict[str, Any] = {}

    with Timer() as timer:
        if validation_report.passed:
            panel_result = build_monthly_panel(raw, cfg)
            write_json(report_dir / "panel_report.json", panel_result.report.as_dict())

            panel_validation = validate_panel(panel_result.panel, cfg)
            write_json(report_dir / "panel_validation.json", panel_validation.as_dict())
            if not panel_validation.passed:
                validation_report.add(
                    CheckResult(
                        name="panel_schema",
                        passed=False,
                        detail="; ".join(
                            check.detail for check in panel_validation.checks if not check.passed
                        ),
                    )
                )
            else:
                metadata = {
                    **panel_result.report.as_dict(),
                    "dataset_version": versions.dataset_version,
                    "dvc_hash": versions.dvc_hash,
                    "git_commit": versions.git_commit,
                    "raw_path": str(raw_path),
                }
                panel_path = save_panel(panel_result.panel, cfg.panel_path, metadata)
                rows_out = len(panel_result.panel)
                panel_start = panel_result.report.start or ""
                panel_end = panel_result.report.end or ""
                if settings.enable_dvc:
                    dvc_result = dvc_add([raw_path, panel_path], PROJECT_ROOT)

        outcome = IngestionOutcome(
            raw_path=str(raw_path),
            panel_path=str(panel_path),
            dataset_version=versions.dataset_version,
            rows_in=profile.n_rows,
            rows_out=rows_out,
            panel_start=panel_start,
            panel_end=panel_end,
            validation_passed=validation_report.passed,
            report_paths={
                "raw_profile": str(report_dir / "raw_profile.json"),
                "validation_report": str(report_dir / "validation_report.json"),
                "panel_report": str(report_dir / "panel_report.json"),
                "panel_validation": str(report_dir / "panel_validation.json"),
            },
            dvc=dvc_result,
        )

        db_persisted = False
        with optional_session(settings) as session:
            if session is not None:
                record_ingestion(
                    session,
                    dataset_name=cfg.name,
                    source=cfg.source,
                    raw_path=str(raw_path),
                    dataset_version=versions.dataset_version,
                    rows_in=profile.n_rows,
                    rows_out=rows_out,
                    validation_passed=validation_report.passed,
                    report=validation_report.as_dict(),
                )
                record_pipeline_execution(
                    session,
                    pipeline="ingestion",
                    status="succeeded" if validation_report.passed else "failed",
                    duration_seconds=timer.seconds,
                    dataset_version=versions.dataset_version,
                    details={"rows_in": profile.n_rows, "rows_out": rows_out},
                )
                db_persisted = True
        outcome = replace(outcome, db_persisted=db_persisted)

        record_pipeline_state(
            settings,
            pipeline="ingestion",
            status="succeeded" if validation_report.passed else "failed",
            duration_seconds=timer.seconds,
            details=outcome.as_dict(),
        )

    final = replace(outcome, duration_seconds=timer.seconds)
    if not final.validation_passed:
        validation_report.raise_if_failed()

    logger.info(
        "ingestion pipeline finished",
        extra={
            "event": "ingestion_complete",
            "dataset_version": final.dataset_version,
            "rows_out": final.rows_out,
        },
    )
    return final


def ensure_panel(data_cfg: DatasetConfig | None = None) -> tuple[Any, dict[str, Any]]:
    """Load the persisted panel, or fail with an actionable message."""
    cfg = data_cfg or load_data_config()
    if not cfg.panel_path.exists():
        raise IngestionError(
            f"processed panel not found at {cfg.panel_path}; run the ingestion "
            "pipeline first (`make ingest` or DAG `ingestion_pipeline`)"
        )
    return load_panel(cfg.panel_path)
