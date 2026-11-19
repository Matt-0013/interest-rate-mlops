"""Health endpoints.

``/health`` is a liveness/status endpoint and always answers 200 so a Kubernetes
liveness probe never restarts a pod because a dependency is down; ``/health/ready``
is the readiness gate that returns 503 when a required dependency is unavailable.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from api.schemas.system import DependencyStatus, HealthResponse
from interest_rate_mlops.exceptions import ModelNotAvailableError

router = APIRouter(tags=["system"])
_STARTED = time.monotonic()


def _health_payload(request: Request) -> tuple[HealthResponse, bool]:
    settings = request.app.state.settings
    database = request.app.state.db
    db_ok, db_detail = database.check() if database is not None else (False, "not configured")

    model_ok = True
    model_detail = "not loaded yet"
    try:
        handle = request.app.state.model_service.get()
        model_detail = f"{handle.name} v{handle.version} ({handle.alias})"
    except ModelNotAvailableError as exc:
        model_ok = False
        model_detail = str(exc)[:300]
    except Exception as exc:  # noqa: BLE001 - health must never raise
        model_ok = False
        model_detail = f"model check failed: {exc}"[:300]

    dependencies = [
        DependencyStatus(name="database", healthy=db_ok, detail=db_detail),
        DependencyStatus(name="model_registry", healthy=model_ok, detail=model_detail),
    ]
    ready = db_ok and model_ok
    return (
        HealthResponse(
            status="ok" if ready else "degraded",
            ready=ready,
            version=settings.api_version,
            environment=settings.app_env,
            uptime_seconds=round(time.monotonic() - _STARTED, 3),
            dependencies=dependencies,
            checks={"database": db_ok, "model": model_ok},
        ),
        ready,
    )


@router.get("/health", response_model=HealthResponse, summary="API health")
def health(request: Request) -> HealthResponse:
    payload, _ = _health_payload(request)
    return payload


@router.get("/health/ready", summary="Readiness probe")
def ready(request: Request):
    payload, is_ready = _health_payload(request)
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
        )
    return payload
