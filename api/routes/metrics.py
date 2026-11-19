"""Prometheus exposition endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["observability"])


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    """Expose the default registry (API + file-backed ML/pipeline metrics)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
