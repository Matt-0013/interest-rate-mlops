"""Domain exceptions used across the platform.

Keeping these separate from generic ``ValueError`` lets the API translate them
into precise HTTP status codes and lets orchestration distinguish a data
problem from an infrastructure problem.
"""

from __future__ import annotations


class MLOpsError(Exception):
    """Base class for all platform errors."""


class ConfigurationError(MLOpsError):
    """A configuration file or environment variable is missing or invalid."""


class DataValidationError(MLOpsError):
    """The dataset failed a structural or schema validation gate."""

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = failures or []


class IngestionError(MLOpsError):
    """The dataset could not be retrieved or parsed."""


class ModelNotAvailableError(MLOpsError):
    """No model is available for serving (registry unreachable or empty)."""


class PromotionError(MLOpsError):
    """A model promotion could not be applied."""


class RegistryError(MLOpsError):
    """MLflow registry interaction failed."""
