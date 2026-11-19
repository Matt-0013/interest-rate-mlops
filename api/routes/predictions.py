"""Stored prediction history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies.database import get_session
from api.schemas.prediction import PredictionListResponse, PredictionRecordOut
from interest_rate_mlops.database.repositories import count_predictions, list_predictions

router = APIRouter(tags=["inference"])


@router.get(
    "/predictions",
    response_model=PredictionListResponse,
    summary="Retrieve previously stored predictions",
)
def get_predictions(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    model_name: str | None = Query(default=None, max_length=64),
    session: Session = Depends(get_session),
) -> PredictionListResponse:
    """Newest first, with a total count for pagination."""
    records = list_predictions(session, limit=limit, offset=offset, model_name=model_name)
    return PredictionListResponse(
        items=[PredictionRecordOut.model_validate(record) for record in records],
        total=count_predictions(session, model_name=model_name),
        limit=limit,
        offset=offset,
    )
