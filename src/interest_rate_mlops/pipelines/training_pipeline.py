"""Model training pipeline (Airflow DAG 2, ``make train``).

Loads the versioned panel, engineers features, trains and cross-validates all
three candidates, writes the comparison report, logs every run to MLflow and
registers the selected candidate under the ``challenger`` alias.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from interest_rate_mlops.config import (
    DatasetConfig,
    PipelineConfig,
    Settings,
    get_settings,
    load_data_config,
    load_pipeline_config,
)
from interest_rate_mlops.database.repositories import (
    record_pipeline_execution,
    record_training_run,
    upsert_model_metadata,
)
from interest_rate_mlops.exceptions import MLOpsError
from interest_rate_mlops.features import build_supervised_features
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.pipelines.common import (
    Timer,
    optional_session,
    record_pipeline_state,
    utc_stamp,
)
from interest_rate_mlops.pipelines.ingestion_pipeline import ensure_panel
from interest_rate_mlops.registry import MLflowTracker, ModelRegistry
from interest_rate_mlops.registry.tracking import LoggedRun
from interest_rate_mlops.storage import write_json, write_parquet
from interest_rate_mlops.training import ModelTrainer, TrainingResult
from interest_rate_mlops.versioning import build_version_record

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingOutcome:
    """Result of one training run."""

    result: TrainingResult
    logged_runs: list[LoggedRun] = field(default_factory=list)
    candidate_version: str | None = None
    tracking_uri: str = ""
    features_path: str = ""
    db_persisted: bool = False
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.result.best_model,
            "dataset_version": self.result.dataset_version,
            "evaluation_date": self.result.evaluation_date,
            "artifacts_dir": str(self.result.artifacts_dir),
            "candidate_version": self.candidate_version,
            "tracking_uri": self.tracking_uri,
            "scores": self.result.scores(),
            "logged_runs": [
                {
                    "model_name": run.model_name,
                    "run_id": run.run_id,
                    "model_uri": run.model_uri,
                    "registered_version": run.registered_version,
                }
                for run in self.logged_runs
            ],
            "db_persisted": self.db_persisted,
            "warnings": list(self.warnings),
            "duration_seconds": round(self.duration_seconds, 3),
        }


def _persist_feature_matrix(matrix: Any, path: Path) -> Path:
    """Persist features + realised target for monitoring and inspection."""
    frame = matrix.X.copy()
    frame["target"] = matrix.y
    frame["target_date"] = pd.to_datetime(matrix.target_dates)
    frame.index.name = "origin_date"
    return write_parquet(frame, path)


def run_training(
    *,
    settings: Settings | None = None,
    data_cfg: DatasetConfig | None = None,
    pipeline_cfg: PipelineConfig | None = None,
    register: bool = True,
    output_dir: str | Path | None = None,
) -> TrainingOutcome:
    """Train, evaluate, log and register the candidate models.

    Args:
        settings: environment settings.
        data_cfg: dataset configuration.
        pipeline_cfg: modelling configuration.
        register: register the selected candidate in the MLflow registry.
        output_dir: override the artifact directory (used by tests).
    """
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    pipeline_cfg = pipeline_cfg or load_pipeline_config()

    panel, panel_metadata = ensure_panel(data_cfg)
    matrix = build_supervised_features(
        panel,
        pipeline_cfg.features,
        target_col=data_cfg.target.slug,
        horizon=data_cfg.target.horizon,
    )

    versions = build_version_record(data_cfg.raw_path)
    # The panel metadata carries the version it was ingested from; surface a
    # mismatch rather than silently training on a stale panel.
    warnings: list[str] = []
    panel_version = panel_metadata.get("dataset_version")
    if panel_version and panel_version != versions.dataset_version:
        warnings.append(
            "panel dataset_version "
            f"{panel_version!r} differs from raw dataset_version "
            f"{versions.dataset_version!r}; re-run ingestion for a consistent lineage"
        )
        logger.warning(warnings[-1], extra={"event": "dataset_version_mismatch"})

    evaluation_dir = (
        Path(output_dir) if output_dir else Path(settings.reports_path) / "evaluation" / utc_stamp()
    )

    with Timer() as timer:
        trainer = ModelTrainer(
            pipeline_cfg,
            target_name=data_cfg.target.slug,
            target_unit=data_cfg.target.unit,
            dataset_name=data_cfg.name,
            dataset_version=versions.dataset_version,
            horizon=data_cfg.target.horizon,
            output_dir=evaluation_dir,
            dvc_hash=versions.dvc_hash,
            git_commit=versions.git_commit,
        )
        training = trainer.train(matrix)

        features_path = _persist_feature_matrix(matrix, data_cfg.features_path)
        write_json(
            Path(training.artifacts_dir) / "run_metadata.json",
            {
                **training.comparison,
                "panel": panel_metadata,
                "warnings": warnings,
            },
        )

        # Stable paths (reports/evaluation/latest) so DVC metrics, the README and
        # reviewers can find the newest comparison without globbing timestamps.
        latest_dir = Path(settings.reports_path) / "evaluation" / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(Path(training.artifacts_dir).iterdir()):
            if artifact.is_file() and artifact.suffix in {".json", ".csv", ".md"}:
                shutil.copy2(artifact, latest_dir / artifact.name)

        tracker = MLflowTracker(
            tracking_uri=settings.resolved_tracking_uri,
            experiment_name=pipeline_cfg.mlflow.experiment_name,
            registered_model_name=pipeline_cfg.mlflow.registered_model_name,
            artifact_path=pipeline_cfg.mlflow.artifact_path,
            artifact_root=settings.mlflow_artifact_root,
        )

        logged: list[LoggedRun] = []
        for model_result in training.models:
            should_register = register and (
                not pipeline_cfg.mlflow.register_best_only
                or model_result.name == training.best_model
            )
            logged.append(
                tracker.log_candidate(
                    model_result,
                    training,
                    pipeline_cfg,
                    register=should_register,
                    extra_tags={"candidate_role": "selected" if should_register else "benchmark"},
                )
            )

        candidate = next((run for run in logged if run.registered_version is not None), None)
        if candidate is not None:
            registry = ModelRegistry(
                settings.resolved_tracking_uri, pipeline_cfg.mlflow.registered_model_name
            )
            registry.set_alias(pipeline_cfg.mlflow.challenger_alias, candidate.registered_version)
        elif register:
            warnings.append("no model version was registered; check the MLflow backend")

        db_persisted = False
        with optional_session(settings) as session:
            if session is not None:
                for run, model_result in zip(logged, training.models, strict=True):
                    record_training_run(
                        session,
                        run_id=run.run_id,
                        model_name=model_result.name,
                        model_type=model_result.model_type,
                        experiment_name=pipeline_cfg.mlflow.experiment_name,
                        dataset_name=data_cfg.name,
                        dataset_version=versions.dataset_version,
                        params=model_result.params,
                        metrics=model_result.flat_metrics(),
                        cv_metrics=model_result.cv_metrics,
                        model_version=str(run.registered_version)
                        if run.registered_version is not None
                        else None,
                        git_commit=versions.git_commit,
                        duration_seconds=model_result.train_seconds,
                    )
                if candidate is not None:
                    upsert_model_metadata(
                        session,
                        model_name=pipeline_cfg.mlflow.registered_model_name,
                        version=str(candidate.registered_version),
                        alias=pipeline_cfg.mlflow.challenger_alias,
                        stage="challenger",
                        metrics=training.best.flat_metrics(),
                        run_id=candidate.run_id,
                        dataset_version=versions.dataset_version,
                        is_active=False,
                    )
                record_pipeline_execution(
                    session,
                    pipeline="training",
                    status="succeeded",
                    duration_seconds=timer.seconds,
                    dataset_version=versions.dataset_version,
                    model_version=str(candidate.registered_version)
                    if candidate is not None
                    else None,
                    details={"best_model": training.best_model},
                )
                db_persisted = True

        record_pipeline_state(
            settings,
            pipeline="training",
            status="succeeded",
            duration_seconds=timer.seconds,
            details={
                "best_model": training.best_model,
                "candidate_version": candidate.registered_version if candidate else None,
                "dataset_version": versions.dataset_version,
            },
        )

    outcome = TrainingOutcome(
        result=training,
        logged_runs=logged,
        candidate_version=str(candidate.registered_version)
        if candidate and candidate.registered_version is not None
        else None,
        tracking_uri=settings.resolved_tracking_uri,
        features_path=str(features_path),
        db_persisted=db_persisted,
        warnings=warnings,
        duration_seconds=timer.seconds,
    )
    if candidate is None and register:
        raise MLOpsError(
            "training completed but no model was registered in the MLflow registry; "
            "the API needs a registered candidate. Check the MLflow backend is database-backed."
        )
    return replace(outcome)
