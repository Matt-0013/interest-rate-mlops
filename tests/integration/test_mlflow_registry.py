"""MLflow integration: log a run, register it, alias it, load it, predict."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import make_panel

from interest_rate_mlops.config import ModelSpec, load_pipeline_config
from interest_rate_mlops.features import build_supervised_features
from interest_rate_mlops.registry import MLflowTracker, ModelRegistry
from interest_rate_mlops.training import ModelTrainer


@pytest.mark.slow
def test_tracking_registry_and_model_loading(tmp_path: Path):
    config = load_pipeline_config().model_copy(
        update={
            "models": {"linear_regression": ModelSpec(type="linear_regression", params={})},
            "mlflow": load_pipeline_config().mlflow.model_copy(
                update={"registered_model_name": "integration_forecaster"}
            ),
        }
    )
    matrix = build_supervised_features(make_panel(60), config.features, target_col="target")
    trainer = ModelTrainer(
        config,
        target_name="target",
        target_unit="percent",
        dataset_name="synthetic",
        dataset_version="sha256:integration",
        output_dir=tmp_path / "evaluation",
    )
    training = trainer.train(matrix)

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name="integration-experiment",
        registered_model_name="integration_forecaster",
        artifact_path="model",
    )
    logged = tracker.log_candidate(
        training.best, training, config, register=True, extra_tags={"test": "integration"}
    )
    assert logged.run_id
    assert logged.registered_version == 1

    registry = ModelRegistry(tracking_uri, "integration_forecaster")
    registry.set_alias("champion", logged.registered_version)

    info = registry.get_version_by_alias("champion")
    assert info is not None
    assert info.version == "1"
    assert info.metrics["test_rmse"] > 0
    assert registry.model_metrics("1")["test_rmse"] == pytest.approx(
        training.best.test_metrics["rmse"]
    )
    assert [version.version for version in registry.list_versions()] == ["1"]

    model = registry.load_model("champion")
    predictions = model.predict(training.matrix.X.head(4))
    assert len(predictions) == 4
    assert all(isinstance(float(value), float) for value in predictions)


@pytest.mark.slow
def test_registry_reports_missing_alias(tmp_path: Path):
    registry = ModelRegistry(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}", "absent_model")
    assert registry.get_version_by_alias("champion") is None
