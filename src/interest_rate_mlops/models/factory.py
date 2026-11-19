"""Estimator construction from configuration.

Hyperparameters live in ``configs/model.yaml``; adding or retuning a model is a
configuration change, never a code change. Every estimator is constructed fresh
for each fit so cross-validation folds cannot leak fitted state.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from interest_rate_mlops.config import ModelSpec
from interest_rate_mlops.exceptions import ConfigurationError

DEFAULT_RANDOM_STATE = 42


def build_estimator(spec: ModelSpec, random_state: int = DEFAULT_RANDOM_STATE) -> RegressorMixin:
    """Instantiate the estimator described by ``spec``.

    Raises:
        ConfigurationError: for an unsupported model type or invalid parameters.
    """
    params: dict[str, Any] = dict(spec.params)

    try:
        if spec.type == "linear_regression":
            return LinearRegression(**params)
        if spec.type == "random_forest":
            params.setdefault("random_state", random_state)
            return RandomForestRegressor(**params)
        if spec.type == "xgboost":
            params.setdefault("random_state", random_state)
            # Single-threaded determinism is not required, but hist is both fast
            # and stable on CPU-only local machines.
            params.setdefault("tree_method", "hist")
            return XGBRegressor(**params)
    except TypeError as exc:
        raise ConfigurationError(
            f"invalid hyperparameters for model type {spec.type!r}: {exc}"
        ) from exc

    raise ConfigurationError(f"unsupported model type: {spec.type!r}")


def clone_estimator(estimator: RegressorMixin) -> RegressorMixin:
    """Clone a fitted estimator so each evaluation sees an unfitted model."""
    return clone(estimator)
