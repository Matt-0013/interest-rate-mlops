"""MLflow experiment tracking.

Each candidate gets its own run so Linear Regression, Random Forest and XGBoost
are directly comparable in the MLflow UI: identical params namespace, identical
metric names, the same dataset/version tags, and the comparison report attached
as an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from interest_rate_mlops.config import PipelineConfig
from interest_rate_mlops.exceptions import RegistryError
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.registry.store import ensure_tracking_store
from interest_rate_mlops.training.trainer import ModelResult, TrainingResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoggedRun:
    """Identifiers produced by one tracking run."""

    model_name: str
    run_id: str
    model_uri: str
    registered_version: int | None


def _flatten_params(prefix: str, params: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(f"{name}.", value))
        else:
            flat[name] = value
    return flat


class MLflowTracker:
    """Thin, explicit wrapper around MLflow tracking + registry writes."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        registered_model_name: str,
        artifact_path: str = "model",
        artifact_root: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name
        self.artifact_path = artifact_path
        self.artifact_root = artifact_root
        ensure_tracking_store(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    # -- experiment --------------------------------------------------------
    def ensure_experiment(self) -> str:
        """Return the experiment id, creating the experiment when absent."""
        existing = self.client.get_experiment_by_name(self.experiment_name)
        if existing is not None:
            return existing.experiment_id
        try:
            artifact_location = (
                Path(self.artifact_root).resolve().as_uri() if self.artifact_root else None
            )
            return self.client.create_experiment(
                self.experiment_name, artifact_location=artifact_location
            )
        except MlflowException as exc:  # pragma: no cover - race/exotic backends
            raise RegistryError(f"could not create MLflow experiment: {exc}") from exc

    # -- logging -----------------------------------------------------------
    def log_candidate(
        self,
        result: ModelResult,
        training: TrainingResult,
        config: PipelineConfig,
        *,
        register: bool,
        extra_tags: dict[str, str] | None = None,
    ) -> LoggedRun:
        """Log one candidate's params, metrics and artifacts to MLflow."""
        experiment_id = self.ensure_experiment()
        matrix = training.matrix

        params: dict[str, Any] = {
            "model_type": result.model_type,
            **{f"hyperparam.{key}": value for key, value in result.params.items()},
            **{f"feature.{key}": value for key, value in config.features.model_dump().items()},
            "split.test_fraction": config.split.test_fraction,
            "split.strategy": config.split.strategy,
            "cv.n_splits": config.cv.n_splits,
            "cv.gap": config.cv.gap,
            "target.name": training.target_name,
            "target.unit": training.target_unit,
            "dataset.name": training.dataset_name,
            "dataset.version": training.dataset_version,
        }
        # MLflow params must be scalars; list params are joined.
        params = {
            key: "|".join(map(str, value)) if isinstance(value, list) else value
            for key, value in _flatten_params("", params).items()
        }

        tags = {
            "model_family": result.model_type,
            "dataset_version": training.dataset_version,
            "dataset_name": training.dataset_name,
            "git_commit": training.git_commit or "unknown",
            "evaluation_date": training.evaluation_date,
            "selection_rule": config.selection.primary_metric,
            "selected_model": training.best_model,
            **(extra_tags or {}),
        }

        try:
            with mlflow.start_run(experiment_id=experiment_id, run_name=result.name) as run:
                mlflow.log_params(params)
                mlflow.log_metrics(result.flat_metrics())
                mlflow.set_tags(tags)
                mlflow.log_dict(matrix.metadata(), "features/feature_metadata.json")
                mlflow.log_dict(
                    {"split": training.split.as_dict(), "cv_folds": result.cv_folds},
                    "evaluation/validation_protocol.json",
                )
                for path in training.report_paths.values():
                    mlflow.log_artifact(path, artifact_path="reports")
                for paths in training.extra_artifacts.values():
                    for path in paths:
                        mlflow.log_artifact(path, artifact_path="plots")

                signature = infer_signature(matrix.X, result.estimator.predict(matrix.X))
                model_info = mlflow.sklearn.log_model(
                    result.estimator,
                    artifact_path=self.artifact_path,
                    signature=signature,
                    input_example=matrix.X.head(3),
                    registered_model_name=self.registered_model_name if register else None,
                )
                run_id = run.info.run_id
        except MlflowException as exc:
            raise RegistryError(f"MLflow logging failed for {result.name}: {exc}") from exc

        version: int | None = None
        if register:
            version = self._find_registered_version(run_id)

        logger.info(
            "logged MLflow run",
            extra={
                "event": "mlflow_run_logged",
                "run_id": run_id,
                "model_name": result.name,
                "model_version": version,
            },
        )
        return LoggedRun(
            model_name=result.name,
            run_id=run_id,
            model_uri=model_info.model_uri,
            registered_version=version,
        )

    def _find_registered_version(self, run_id: str) -> int:
        """Resolve the model version created for ``run_id`` (API-version tolerant)."""
        try:
            versions = self.client.search_model_versions(f"name = '{self.registered_model_name}'")
        except MlflowException as exc:
            raise RegistryError(f"could not search model versions: {exc}") from exc
        matching = [int(version.version) for version in versions if version.run_id == run_id]
        if not matching:
            raise RegistryError(
                f"no registered model version found for run {run_id}; "
                "check that the MLflow backend is a database (registry requires it)"
            )
        return max(matching)
