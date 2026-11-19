"""End-to-end pipeline check used by CI (small fixture, local MLflow).

This is the test that proves the pipeline produces real artifacts: a validated
panel, three trained models, a comparison report, a registered candidate, a
promotion decision and a monitoring run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interest_rate_mlops.config import (
    ModelSpec,
    MonitoringPaths,
    Settings,
    load_monitoring_config,
)
from interest_rate_mlops.pipelines import run_ingestion, run_monitoring, run_promotion, run_training


def _cheap_models() -> dict[str, ModelSpec]:
    return {
        "linear_regression": ModelSpec(type="linear_regression", params={}),
        "random_forest": ModelSpec(
            type="random_forest",
            params={"n_estimators": 20, "max_depth": 4, "random_state": 42, "n_jobs": 1},
        ),
        "xgboost": ModelSpec(
            type="xgboost",
            params={
                "n_estimators": 25,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42,
                "n_jobs": 1,
                "tree_method": "hist",
            },
        ),
    }


@pytest.mark.slow
def test_full_pipeline_on_fixture(tmp_path: Path, dataset_config, pipeline_config):
    settings = Settings(
        app_env="test",
        enable_db_persistence=False,
        enable_dvc=False,
        reports_dir=str(tmp_path / "reports"),
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        log_format="console",
    )
    data_cfg = dataset_config.model_copy(update={"processed_dir": str(tmp_path / "processed")})
    config = pipeline_config.model_copy(
        update={
            "models": _cheap_models(),
            "mlflow": pipeline_config.mlflow.model_copy(
                update={
                    "registered_model_name": "ci_forecaster",
                    "experiment_name": "ci-experiment",
                }
            ),
        }
    )

    # 1. Ingestion with validation.
    ingestion = run_ingestion(settings=settings, data_cfg=data_cfg)
    assert ingestion.validation_passed
    assert ingestion.rows_in == 218
    assert ingestion.rows_out == 216
    assert ingestion.dataset_version.startswith("sha256:")
    assert Path(ingestion.panel_path).exists()

    # 2. Training across all three models.
    training = run_training(settings=settings, data_cfg=data_cfg, pipeline_cfg=config)
    assert len(training.result.models) == 3
    assert training.candidate_version == "1"
    assert Path(training.features_path).exists()

    comparison = json.loads(Path(training.result.report_paths["json"]).read_text(encoding="utf-8"))
    assert comparison["selection"]["selected_model"] == training.result.best_model
    assert len(comparison["models"]) == 3
    assert all(model["test_rmse"] is not None for model in comparison["models"])
    assert Path(training.result.report_paths["markdown"]).exists()

    # 3. Promotion with gates (bootstrap path: no incumbent yet).
    promotion = run_promotion(settings=settings, pipeline_cfg=config)
    assert promotion.promoted
    assert promotion.candidate is not None
    assert promotion.candidate.version == "1"
    assert Path(promotion.report_path).exists()

    # 4. Monitoring over the persisted feature matrix.
    monitoring_cfg = load_monitoring_config().model_copy(
        update={
            "paths": MonitoringPaths(
                reports_dir=str(tmp_path / "reports" / "monitoring"),
                status_file=str(tmp_path / "reports" / "drift_status.json"),
                latest_summary=str(tmp_path / "reports" / "monitoring" / "latest_summary.json"),
            )
        }
    )
    monitoring = run_monitoring(
        settings=settings,
        data_cfg=data_cfg,
        pipeline_cfg=config,
        monitoring_cfg=monitoring_cfg,
    )
    assert monitoring.status in {"ok", "warning", "critical"}
    assert Path(monitoring.status_path).exists()
    assert monitoring.drift.n_current_rows > 0
    # Drift and performance are reported independently.
    assert monitoring.performance.status in {"ok", "warning", "critical", "unknown"}


@pytest.mark.slow
def test_training_requires_a_panel(tmp_path: Path, dataset_config, pipeline_config):
    settings = Settings(
        app_env="test",
        enable_db_persistence=False,
        enable_dvc=False,
        reports_dir=str(tmp_path / "reports"),
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        log_format="console",
    )
    data_cfg = dataset_config.model_copy(update={"processed_dir": str(tmp_path / "empty")})
    with pytest.raises(Exception, match="run the ingestion pipeline"):
        run_training(settings=settings, data_cfg=data_cfg, pipeline_cfg=pipeline_config)
