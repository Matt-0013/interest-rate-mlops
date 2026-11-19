"""Estimator factory.

One factory for all candidates keeps the training loop free of per-algorithm
branches, which is what makes the three models comparable by construction.
"""

from interest_rate_mlops.models.factory import build_estimator, clone_estimator

__all__ = ["build_estimator", "clone_estimator"]
