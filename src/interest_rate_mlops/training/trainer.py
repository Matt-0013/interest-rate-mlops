"""Training loop shared by Linear Regression, Random Forest and XGBoost.

Guarantees that make the comparison fair:

* identical design matrix, split and CV folds for every candidate;
* a freshly constructed estimator per fit (no state carried between folds);
* identical metric code for every model;
* the winner is chosen on cross-validated error, and the chronological test
  block is reported but never used for selection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from interest_rate_mlops.config import PipelineConfig, SelectionConfig
from interest_rate_mlops.evaluation.metrics import regression_metrics
from interest_rate_mlops.evaluation.plots import (
    plot_model_comparison,
    plot_training_timeline,
    save_all_diagnostics,
)
from interest_rate_mlops.evaluation.report import build_comparison, save_comparison, utc_timestamp
from interest_rate_mlops.features.builders import FeatureMatrix
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.models.factory import build_estimator
from interest_rate_mlops.training.splits import Split, chronological_split, walk_forward_splits

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """One trained candidate with its predictions and scores."""

    name: str
    model_type: str
    params: dict[str, Any]
    estimator: Any
    y_true: pd.Series
    predictions: pd.Series
    residuals: pd.Series
    test_metrics: dict[str, float | None]
    cv_metrics: dict[str, float | None]
    cv_folds: list[dict[str, float | None]]
    train_seconds: float
    predict_seconds: float
    n_train: int
    n_test: int

    @property
    def cv_rmse_std_ratio(self) -> float | None:
        mean = self.cv_metrics.get("rmse")
        std = self.cv_metrics.get("rmse_std")
        if mean in (None, 0) or std is None:
            return None
        return float(std) / float(mean)

    def summary(self) -> dict[str, Any]:
        """Flat, report-ready description of this candidate."""
        return {
            "model_name": self.name,
            "model_type": self.model_type,
            "hyperparameters": self.params,
            "test_mae": self.test_metrics.get("mae"),
            "test_rmse": self.test_metrics.get("rmse"),
            "test_mape": self.test_metrics.get("mape"),
            "test_mape_coverage_pct": self.test_metrics.get("mape_coverage_pct"),
            "cv_rmse_mean": self.cv_metrics.get("rmse"),
            "cv_mae_mean": self.cv_metrics.get("mae"),
            "cv_rmse_std": self.cv_metrics.get("rmse_std"),
            "cv_rmse_std_ratio": self.cv_rmse_std_ratio,
            "train_duration_seconds": round(self.train_seconds, 4),
            "predict_duration_seconds": round(self.predict_seconds, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "cv_folds": self.cv_folds,
        }

    def flat_metrics(self) -> dict[str, float]:
        """MLflow-ready numeric metrics (``None`` values removed)."""
        candidates = {
            "test_mae": self.test_metrics.get("mae"),
            "test_rmse": self.test_metrics.get("rmse"),
            "test_mape": self.test_metrics.get("mape"),
            "test_mape_coverage_pct": self.test_metrics.get("mape_coverage_pct"),
            "cv_rmse_mean": self.cv_metrics.get("rmse"),
            "cv_mae_mean": self.cv_metrics.get("mae"),
            "cv_mape_mean": self.cv_metrics.get("mape"),
            "cv_rmse_std": self.cv_metrics.get("rmse_std"),
            "cv_rmse_std_ratio": self.cv_rmse_std_ratio,
            "train_duration_seconds": self.train_seconds,
            "predict_duration_seconds": self.predict_seconds,
        }
        return {
            key: float(value)
            for key, value in candidates.items()
            if value is not None and np.isfinite(value)
        }


@dataclass
class TrainingResult:
    """Everything a downstream stage (tracking, registry, API) needs."""

    models: list[ModelResult]
    best_model: str
    split: Split
    matrix: FeatureMatrix
    comparison: dict[str, Any]
    report_paths: dict[str, str]
    artifacts_dir: Path
    dataset_name: str
    dataset_version: str
    target_name: str
    target_unit: str
    evaluation_date: str
    feature_metadata: dict[str, Any]
    dvc_hash: str | None = None
    git_commit: str | None = None
    extra_artifacts: dict[str, list[str]] = field(default_factory=dict)

    def by_name(self, name: str) -> ModelResult:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(f"unknown model {name!r}")

    @property
    def best(self) -> ModelResult:
        return self.by_name(self.best_model)

    def scores(self) -> dict[str, dict[str, Any]]:
        return {model.name: model.summary() for model in self.models}


def aggregate_fold_metrics(folds: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Mean of each metric across folds, plus the standard deviation of RMSE."""
    if not folds:
        return {}

    aggregated: dict[str, float | None] = {}
    keys = set().union(*(fold.keys() for fold in folds))
    for key in keys:
        values = [fold[key] for fold in folds if fold.get(key) is not None]
        aggregated[key] = float(np.mean(values)) if values else None

    rmse_values = [fold["rmse"] for fold in folds if fold.get("rmse") is not None]
    aggregated["rmse_std"] = float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0
    aggregated["n_folds"] = float(len(folds))
    return aggregated


def select_best_model(results: list[ModelResult], selection: SelectionConfig) -> str:
    """Pick the winner by CV score, with deterministic tie-breaking."""

    def score(result: ModelResult, metric: str) -> float:
        if metric.startswith("cv_"):
            value = result.cv_metrics.get(metric.removeprefix("cv_"))
        else:
            value = result.test_metrics.get(metric.removeprefix("test_"))
        return float(value) if value is not None else float("inf")

    direction = 1.0 if selection.minimize else -1.0
    ranked = sorted(
        results,
        key=lambda item: (
            direction * score(item, selection.primary_metric),
            direction * score(item, selection.secondary_metric),
            item.name,
        ),
    )
    return ranked[0].name


class ModelTrainer:
    """Trains and evaluates every configured candidate on one feature matrix."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        target_name: str,
        target_unit: str,
        dataset_name: str,
        dataset_version: str,
        horizon: int = 1,
        output_dir: str | Path | None = None,
        random_state: int = 42,
        dvc_hash: str | None = None,
        git_commit: str | None = None,
    ) -> None:
        self.config = config
        self.target_name = target_name
        self.target_unit = target_unit
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.horizon = horizon
        self.output_dir = Path(output_dir) if output_dir else None
        self.random_state = random_state
        self.dvc_hash = dvc_hash
        self.git_commit = git_commit

    # -- internals ----------------------------------------------------------
    def _cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        spec_type: str,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[dict[str, float | None]]:
        fold_metrics: list[dict[str, float | None]] = []
        for train_index, test_index in folds:
            estimator = build_estimator(self.config.models[spec_type], self.random_state)
            estimator.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            predictions = estimator.predict(X_train.iloc[test_index])
            fold_metrics.append(regression_metrics(y_train.iloc[test_index], predictions))
        return fold_metrics

    def _train_one(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> ModelResult:
        spec = self.config.models[name]

        started = time.perf_counter()
        fold_metrics = self._cross_validate(X_train, y_train, name, folds)
        cv_metrics = aggregate_fold_metrics(fold_metrics)

        estimator = build_estimator(spec, self.random_state)
        fit_started = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        raw_predictions = estimator.predict(X_test)
        predict_seconds = time.perf_counter() - predict_started

        predictions = pd.Series(raw_predictions, index=X_test.index, name="prediction")
        residuals = (y_test - predictions).rename("residual")
        test_metrics = regression_metrics(y_test, raw_predictions)

        logger.info(
            "trained candidate model",
            extra={
                "event": "model_trained",
                "model_name": name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "status_code": 200,
            },
        )
        return ModelResult(
            name=name,
            model_type=spec.type,
            params=dict(spec.params),
            estimator=estimator,
            y_true=y_test,
            predictions=predictions,
            residuals=residuals,
            test_metrics=test_metrics,
            cv_metrics=cv_metrics,
            cv_folds=fold_metrics,
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
        )

    # -- public API ---------------------------------------------------------
    def train(self, matrix: FeatureMatrix) -> TrainingResult:
        """Run the full training/comparison cycle for one feature matrix."""
        if matrix.feature_names != self.config.features.names():
            raise ValueError(
                "feature matrix does not match the configured feature contract:\n"
                f"matrix={matrix.feature_names}\nconfig={self.config.features.names()}"
            )

        split = chronological_split(matrix.origins, self.config.split)
        X, y = matrix.X, matrix.y
        X_train, y_train = X.iloc[split.train_idx], y.iloc[split.train_idx]
        X_test, y_test = X.iloc[split.test_idx], y.iloc[split.test_idx]

        folds = walk_forward_splits(len(X_train), self.config.cv)

        results = [
            self._train_one(name, X_train, y_train, X_test, y_test, folds)
            for name in self.config.models
        ]
        best_model = select_best_model(results, self.config.selection)

        evaluation_date = utc_timestamp()
        directory = self.output_dir or (Path("reports") / "evaluation" / evaluation_date)
        directory.mkdir(parents=True, exist_ok=True)

        diagnostics = save_all_diagnostics(results, directory, unit=self.target_unit)
        plot_model_comparison(
            {result.name: result.test_metrics for result in results},
            directory / "model_comparison.png",
        )
        plot_training_timeline(matrix.origins, directory / "origin_timeline.png")

        comparison = build_comparison(
            [result.summary() for result in results],
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            target_name=self.target_name,
            target_unit=self.target_unit,
            horizon=self.horizon,
            split=split,
            feature_metadata=matrix.metadata(),
            selection=self.config.selection,
            selected_model=best_model,
            evaluation_date=evaluation_date,
            cv_splits=self.config.cv.n_splits,
            dvc_hash=self.dvc_hash,
            git_commit=self.git_commit,
        )
        report_paths = save_comparison(comparison, directory)

        return TrainingResult(
            models=results,
            best_model=best_model,
            split=split,
            matrix=matrix,
            comparison=comparison,
            report_paths=report_paths,
            artifacts_dir=directory,
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            target_name=self.target_name,
            target_unit=self.target_unit,
            evaluation_date=evaluation_date,
            feature_metadata=matrix.metadata(),
            dvc_hash=self.dvc_hash,
            git_commit=self.git_commit,
            extra_artifacts=diagnostics,
        )
