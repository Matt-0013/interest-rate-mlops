"""Champion model metadata."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies.model import ModelHandle, ModelService, get_model_service
from api.schemas.model import ModelInfoResponse
from api.services import latest_feature_row

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse, summary="Approved model name and version")
def model_info(
    model_service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    handle: ModelHandle = model_service.get()
    return ModelInfoResponse(
        name=handle.name,
        version=handle.version,
        alias=handle.alias,
        run_id=handle.run_id,
        dataset_version=handle.dataset_version,
        target=handle.target,
        unit=handle.unit,
        horizon=handle.horizon,
        feature_names=handle.feature_names,
        lookback_observations=handle.lookback_observations,
        metrics=handle.metrics,
        loaded_at=handle.loaded_at,
        cache_ttl_seconds=model_service.settings.model_cache_ttl_seconds,
        tracking_uri=handle.tracking_uri,
        notes=handle.notes,
    )


@router.get("/features", summary="Freshest engineered feature row")
def latest_features(
    model_service: ModelService = Depends(get_model_service),
) -> dict:
    """Return the most recent complete feature vector for use with ``POST /predict``."""
    return latest_feature_row(model_service.get())
