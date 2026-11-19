"""Champion model cache.

Loading a model from MLflow on every request would dominate latency and hammer
the tracking server, so the champion is loaded once, cached for a TTL, and
refreshed under a lock. The feature contract is taken from the model's own
signature first (authoritative), then from the logged feature artifact, and only
then from configuration - with a note explaining any fallback.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import mlflow.artifacts
from fastapi import Request

from api import metrics
from interest_rate_mlops.config import (
    Settings,
    load_data_config,
    load_pipeline_config,
)
from interest_rate_mlops.exceptions import ModelNotAvailableError
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.registry import ModelRegistry

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelHandle:
    """Everything the API needs to know about the served model."""

    model: Any
    name: str
    version: str
    alias: str
    run_id: str | None
    dataset_version: str | None
    feature_names: list[str]
    target: str
    unit: str
    horizon: int
    lookback_observations: int
    metrics: dict[str, float]
    loaded_at: datetime
    tracking_uri: str
    notes: list[str] = field(default_factory=list)


class ModelService:
    """Thread-safe, TTL-bounded champion model provider."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._handle: ModelHandle | None = None
        self._loaded_monotonic: float = 0.0
        self.load_failures = 0

    # -- public ------------------------------------------------------------
    def get(self, *, refresh: bool = False) -> ModelHandle:
        """Return the cached champion, reloading when stale or forced."""
        with self._lock:
            if self._handle is not None and not refresh:
                age = time.monotonic() - self._loaded_monotonic
                if age < self.settings.model_cache_ttl_seconds:
                    return self._handle
            return self._load()

    def invalidate(self) -> None:
        with self._lock:
            self._handle = None
            self._loaded_monotonic = 0.0

    # -- internals ---------------------------------------------------------
    def _load(self) -> ModelHandle:
        pipeline_cfg = load_pipeline_config()
        data_cfg = load_data_config()
        alias = self.settings.model_alias
        registry = ModelRegistry(self.settings.resolved_tracking_uri, self.settings.model_name)

        try:
            info = registry.get_version_by_alias(alias)
            if info is None:
                raise ModelNotAvailableError(
                    f"no model version carries the {alias!r} alias for "
                    f"{self.settings.model_name!r}"
                )
            model = registry.load_model(alias)
        except Exception:
            self.load_failures += 1
            metrics.MODEL_LOAD_FAILURES.inc()
            raise

        notes: list[str] = []
        feature_names = self._feature_names_from_signature(model, notes)
        if not feature_names:
            feature_names = self._feature_names_from_artifact(info.run_id, notes)
        if not feature_names:
            feature_names = list(pipeline_cfg.features.names())
            notes.append(
                "feature contract fell back to configs/model.yaml because neither the "
                "model signature nor the logged feature artifact was readable"
            )

        configured = list(pipeline_cfg.features.names())
        if feature_names != configured:
            notes.append(
                "served feature contract differs from configs/model.yaml; the model's "
                "own contract is used"
            )

        handle = ModelHandle(
            model=model,
            name=self.settings.model_name,
            version=info.version,
            alias=alias,
            run_id=info.run_id,
            dataset_version=info.tags.get("dataset_version"),
            feature_names=feature_names,
            target=data_cfg.target.slug,
            unit=data_cfg.target.unit,
            horizon=data_cfg.target.horizon,
            lookback_observations=pipeline_cfg.features.lookback,
            metrics=info.metrics,
            loaded_at=datetime.now(UTC),
            tracking_uri=self.settings.resolved_tracking_uri,
            notes=notes,
        )

        metrics.MODEL_INFO.clear()
        metrics.MODEL_INFO.labels(handle.name, handle.version, handle.alias).set(1.0)

        self._handle = handle
        self._loaded_monotonic = time.monotonic()
        logger.info(
            "champion model loaded",
            extra={
                "event": "model_loaded",
                "model_name": handle.name,
                "model_version": handle.version,
            },
        )
        return handle

    @staticmethod
    def _feature_names_from_signature(model: Any, notes: list[str]) -> list[str]:
        try:
            signature = getattr(getattr(model, "metadata", None), "signature", None)
            if signature is None or signature.inputs is None:
                return []
            names = list(signature.inputs.input_names())
            return [str(name) for name in names]
        except Exception as exc:  # noqa: BLE001 - optional metadata path
            notes.append(f"model signature could not be read: {exc}")
            return []

    @staticmethod
    def _feature_names_from_artifact(run_id: str | None, notes: list[str]) -> list[str]:
        if not run_id:
            return []
        try:
            payload = mlflow.artifacts.load_dict(f"runs:/{run_id}/features/feature_metadata.json")
            names = payload.get("feature_names") if isinstance(payload, dict) else None
            return [str(name) for name in names] if names else []
        except Exception as exc:  # noqa: BLE001 - artifact may be unavailable
            notes.append(f"logged feature artifact could not be read: {exc}")
            return []


def get_model_service(request: Request) -> ModelService:
    """Resolve the model service from application state."""
    return request.app.state.model_service
