"""Training pipeline behaviour on a small synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import make_panel

from interest_rate_mlops.config import ModelSpec, load_pipeline_config
from interest_rate_mlops.features import build_supervised_features
from interest_rate_mlops.training import ModelTrainer, select_best_model
from interest_rate_mlops.training.trainer import aggregate_fold_metrics


def fast_pipeline_config():
    """Configured pipeline with cheap hyperparameters for tests."""
    config = load_pipeline_config()
    return config.model_copy(
        update={
            "models": {
                "linear_regression": ModelSpec(type="linear_regression", params={}),
                "random_forest": ModelSpec(
                    type="random_forest",
                    params={"n_estimators": 25, "max_depth": 4, "random_state": 42, "n_jobs": 1},
                ),
                "xgboost": ModelSpec(
                    type="xgboost",
                    params={
                        "n_estimators": 30,
                        "max_depth": 3,
                        "learning_rate": 0.1,
                        "random_state": 42,
                        "n_jobs": 1,
                        "tree_method": "hist",
                    },
                ),
            }
        }
    )


@pytest.fixture()
def trained(tmp_path: Path):
    config = fast_pipeline_config()
    panel = make_panel(90)
    matrix = build_supervised_features(panel, config.features, target_col="target")
    trainer = ModelTrainer(
        config,
        target_name="target",
        target_unit="percent",
        dataset_name="synthetic",
        dataset_version="sha256:test",
        output_dir=tmp_path / "evaluation",
    )
    return trainer.train(matrix)


def test_all_three_models_are_trained_and_scored(trained):
    assert [result.name for result in trained.models] == [
        "linear_regression",
        "random_forest",
        "xgboost",
    ]
    for result in trained.models:
        assert np.isfinite(result.test_metrics["rmse"])
        assert np.isfinite(result.test_metrics["mae"])
        assert result.n_train + result.n_test == len(trained.matrix)
        assert result.train_seconds >= 0


def test_models_share_split_and_cv_folds(trained):
    n_train = {result.n_train for result in trained.models}
    assert len(n_train) == 1
    assert trained.split.n_train == n_train.pop()
    fold_counts = {len(result.cv_folds) for result in trained.models}
    assert fold_counts == {trained.comparison["validation"]["n_splits"]}


def test_predictions_align_with_test_origins(trained):
    for result in trained.models:
        assert len(result.predictions) == result.n_test
        assert result.predictions.index.equals(result.y_true.index)
        assert np.isfinite(result.predictions.to_numpy()).all()
        assert np.allclose(
            result.residuals.to_numpy(),
            result.y_true.to_numpy() - result.predictions.to_numpy(),
        )


def test_selection_uses_cross_validated_score(trained):
    best_cv = min(result.cv_metrics["rmse"] for result in trained.models)
    assert trained.by_name(trained.best_model).cv_metrics["rmse"] == pytest.approx(best_cv)
    assert trained.comparison["selection"]["selected_model"] == trained.best_model


def test_comparison_report_and_plots_are_written(trained):
    directory = trained.artifacts_dir
    assert (directory / "model_comparison.json").exists()
    assert (directory / "model_comparison.csv").exists()
    assert (directory / "model_comparison.md").exists()

    payload = json.loads((directory / "model_comparison.json").read_text(encoding="utf-8"))
    assert payload["dataset"]["version"] == "sha256:test"
    assert payload["target"]["unit"] == "percent"
    assert len(payload["models"]) == 3
    assert payload["selection"]["selected_model"] == trained.best_model
    assert payload["models"][0]["hyperparameters"] is not None

    for name in ("linear_regression", "random_forest", "xgboost"):
        assert (directory / f"{name}_actual_vs_predicted.png").exists()
        assert (directory / f"{name}_residuals.png").exists()
        assert (directory / f"{name}_error_distribution.png").exists()
    assert (directory / "model_comparison.png").exists()


def test_training_is_reproducible(tmp_path):
    config = fast_pipeline_config()
    matrix = build_supervised_features(make_panel(90), config.features, target_col="target")
    kwargs = {
        "target_name": "target",
        "target_unit": "percent",
        "dataset_name": "synthetic",
        "dataset_version": "sha256:test",
    }
    first = ModelTrainer(config, output_dir=tmp_path / "run1", **kwargs).train(matrix)
    second = ModelTrainer(config, output_dir=tmp_path / "run2", **kwargs).train(matrix)

    for name in config.models:
        assert first.by_name(name).test_metrics["rmse"] == pytest.approx(
            second.by_name(name).test_metrics["rmse"], rel=1e-9
        )


def test_feature_contract_mismatch_is_rejected(tmp_path):
    config = fast_pipeline_config()
    matrix = build_supervised_features(make_panel(90), config.features, target_col="target")
    narrowed = config.model_copy(
        update={"features": config.features.model_copy(update={"lags": [1, 2]})}
    )
    trainer = ModelTrainer(
        narrowed,
        target_name="target",
        target_unit="percent",
        dataset_name="synthetic",
        dataset_version="sha256:test",
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="feature contract"):
        trainer.train(matrix)


def test_aggregate_fold_metrics_handles_missing_values():
    folds = [
        {"mae": 1.0, "rmse": 2.0, "mape": None},
        {"mae": 3.0, "rmse": 4.0, "mape": 10.0},
    ]
    aggregated = aggregate_fold_metrics(folds)
    assert aggregated["mae"] == pytest.approx(2.0)
    assert aggregated["mape"] == pytest.approx(10.0)
    assert aggregated["rmse_std"] == pytest.approx(np.std([2.0, 4.0], ddof=1))


def test_select_best_model_tie_breaks_deterministically():
    class Fake:
        def __init__(self, name, cv_rmse, cv_mae):
            self.name = name
            self.cv_metrics = {"rmse": cv_rmse, "mae": cv_mae}
            self.test_metrics = {"rmse": cv_rmse, "mae": cv_mae}

    selection = load_pipeline_config().selection
    results = [Fake("b", 1.0, 0.5), Fake("a", 1.0, 0.5), Fake("c", 0.9, 0.9)]
    assert select_best_model(results, selection) == "c"
    assert select_best_model([Fake("b", 1.0, 0.5), Fake("a", 1.0, 0.5)], selection) == "a"
