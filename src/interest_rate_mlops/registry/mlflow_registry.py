"""MLflow Model Registry access.

Promotion uses registry **aliases** (``champion`` / ``challenger``) rather than
the deprecated stage API: an alias is a mutable pointer to an immutable model
version, which is exactly the promotion semantic needed here and keeps full
version history intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from interest_rate_mlops.exceptions import ModelNotAvailableError, RegistryError
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.registry.store import ensure_tracking_store

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegisteredModelInfo:
    """Snapshot of one registry version, safe to serialise."""

    name: str
    version: str
    run_id: str
    aliases: list[str] = field(default_factory=list)
    creation_timestamp: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "aliases": list(self.aliases),
            "creation_timestamp": self.creation_timestamp,
            "metrics": dict(self.metrics),
            "tags": dict(self.tags),
            "source": self.source,
        }


class ModelRegistry:
    """Small, testable facade over ``MlflowClient`` + ``mlflow.pyfunc``."""

    def __init__(self, tracking_uri: str, model_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        ensure_tracking_store(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    # -- reads -------------------------------------------------------------
    def get_version_by_alias(self, alias: str) -> RegisteredModelInfo | None:
        """Registered version currently holding ``alias``, if any."""
        try:
            version = self.client.get_model_version_by_alias(self.model_name, alias)
        except MlflowException:
            return None
        return self._to_info(version)

    def get_version(self, version: str | int) -> RegisteredModelInfo:
        try:
            model_version = self.client.get_model_version(self.model_name, str(version))
        except MlflowException as exc:
            raise RegistryError(
                f"model {self.model_name!r} version {version} not found: {exc}"
            ) from exc
        return self._to_info(model_version)

    def list_versions(self) -> list[RegisteredModelInfo]:
        try:
            versions = self.client.search_model_versions(f"name = '{self.model_name}'")
        except MlflowException as exc:
            raise RegistryError(f"could not list model versions: {exc}") from exc
        return sorted(
            (self._to_info(version) for version in versions), key=lambda v: int(v.version)
        )

    def model_metrics(self, version: str | int) -> dict[str, float]:
        """Training metrics of the run that produced ``version``."""
        info = self.get_version(version)
        try:
            run = self.client.get_run(info.run_id)
        except MlflowException as exc:
            raise RegistryError(f"run {info.run_id} not found: {exc}") from exc
        return {key: float(value) for key, value in run.data.metrics.items()}

    # -- writes ------------------------------------------------------------
    def set_alias(self, alias: str, version: str | int) -> None:
        """Point ``alias`` at ``version`` (atomic in the MLflow backend)."""
        try:
            self.client.set_registered_model_alias(self.model_name, alias, str(version))
        except MlflowException as exc:
            raise RegistryError(f"could not set alias {alias!r}: {exc}") from exc
        logger.info(
            "registry alias updated",
            extra={
                "event": "alias_set",
                "model_name": self.model_name,
                "model_version": str(version),
                "task": alias,
            },
        )

    def tag_version(self, version: str | int, key: str, value: str) -> None:
        try:
            self.client.set_model_version_tag(self.model_name, str(version), key, value)
        except MlflowException as exc:
            raise RegistryError(f"could not tag model version: {exc}") from exc

    def describe_version(self, version: str | int) -> None:
        try:
            self.client.update_model_version(
                self.model_name, str(version), description=f"Registered version {version}"
            )
        except MlflowException:  # pragma: no cover - cosmetic only
            logger.warning("could not update model version description")

    # -- serving -----------------------------------------------------------
    def load_model(self, alias: str) -> Any:
        """Load the pyfunc model behind ``alias``.

        Raises:
            ModelNotAvailableError: when no version carries the alias or the
                tracking server cannot be reached.
        """
        uri = f"models:/{self.model_name}@{alias}"
        try:
            return mlflow.pyfunc.load_model(uri)
        except MlflowException as exc:
            raise ModelNotAvailableError(
                f"could not load model {uri}: {exc}. Is the champion promoted and "
                "the MLflow tracking server reachable?"
            ) from exc

    def _to_info(self, version: Any) -> RegisteredModelInfo:
        run_metrics: dict[str, float] = {}
        try:
            run = self.client.get_run(version.run_id)
            run_metrics = {key: float(value) for key, value in run.data.metrics.items()}
        except MlflowException:  # run may be garbage-collected
            run_metrics = {}
        return RegisteredModelInfo(
            name=version.name,
            version=str(version.version),
            run_id=version.run_id,
            aliases=list(getattr(version, "aliases", []) or []),
            creation_timestamp=getattr(version, "creation_timestamp", None),
            metrics=run_metrics,
            tags=dict(getattr(version, "tags", {}) or {}),
            source=getattr(version, "source", None),
        )
