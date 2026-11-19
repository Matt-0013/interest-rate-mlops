"""Monitoring status endpoint (database first, status file as fallback)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy.orm import Session

from api.schemas.system import DriftStatusResponse
from api.services import read_drift_status

router = APIRouter(tags=["observability"])


@router.get(
    "/drift/status",
    response_model=DriftStatusResponse,
    summary="Latest data drift and performance monitoring status",
)
def drift_status(request: Request) -> DriftStatusResponse:
    """Return the most recent monitor run.

    Data drift and performance degradation are separate fields: drift describes
    input distributions, performance describes realized accuracy.
    """
    database = request.app.state.db
    session: Session | None = None
    if database is not None and database.session_factory is not None:
        try:
            session = database.session_factory()
        except Exception:  # noqa: BLE001 - fall back to the status file
            session = None
    try:
        return read_drift_status(session, request.app.state.settings)
    finally:
        if session is not None:
            session.close()
