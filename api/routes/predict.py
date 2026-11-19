"""Inference endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from api.dependencies.database import get_session
from api.dependencies.model import ModelService, get_model_service
from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.services import perform_prediction

router = APIRouter(tags=["inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Forecast the next monthly interest-rate observation",
    responses={
        422: {"description": "Invalid features/history or a violated feature contract"},
        503: {"description": "No champion model or no database available"},
    },
)
def predict(
    payload: PredictionRequest,
    request: Request,
    model_service: ModelService = Depends(get_model_service),
    session: Session = Depends(get_session),
) -> PredictionResponse:
    """Score one observation using the champion model and persist the result."""
    handle = model_service.get()
    return perform_prediction(
        payload,
        handle,
        session,
        request.app.state.settings,
        request_id=getattr(request.state, "request_id", None),
        source="api",
    )
