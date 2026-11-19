"""FastAPI application.

Run locally with ``uvicorn api.main:app --reload`` or
``make api``. Swagger UI is served at ``/docs`` and GraphiQL at ``/graphql``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from prometheus_client import REGISTRY

from api import metrics  # noqa: F401 - importing registers the metric definitions
from api.dependencies.database import Database
from api.dependencies.model import ModelService
from api.metrics import FileStateCollector
from api.middleware import REQUEST_ID_HEADER, RequestInstrumentationMiddleware
from api.routes import ROUTERS
from api.routes.graphql import router as graphql_router
from interest_rate_mlops.config import Settings, get_settings
from interest_rate_mlops.exceptions import DataValidationError, MLOpsError, ModelNotAvailableError
from interest_rate_mlops.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Forecasting service for the US effective federal funds rate.

* **Approved model** is loaded from the MLflow registry (``champion`` alias) and cached.
* **REST** endpoints cover health, model info, inference, history, metrics and drift.
* **GraphQL** at ``/graphql`` exposes the same operations for typed clients.

Train/serve consistency: the ``features`` mode requires the model's exact feature
contract (``GET /model/info`` lists it); the ``history`` mode accepts a contiguous
monthly series and reuses the training feature code.
"""

_COLLECTOR_REGISTERED = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the database engine and model cache, then tear them down."""
    settings: Settings = app.state.settings
    if app.state.db is None:
        app.state.db = Database(settings)
    if app.state.db.session_factory is None:
        app.state.db.connect()
    if app.state.model_service is None:
        app.state.model_service = ModelService(settings)

    logger.info(
        "api started",
        extra={"event": "api_start", "status_code": status.HTTP_200_OK},
    )
    try:
        yield
    finally:
        app.state.db.dispose()
        logger.info("api stopped", extra={"event": "api_stop"})


def _register_state_collector(settings: Settings) -> None:
    """Expose pipeline/monitoring JSON state through Prometheus (once)."""
    global _COLLECTOR_REGISTERED
    if _COLLECTOR_REGISTERED:
        return
    REGISTRY.register(FileStateCollector(settings.reports_path))
    _COLLECTOR_REGISTERED = True


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None) or request.headers.get(REQUEST_ID_HEADER)


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    model_service: Any | None = None,
) -> FastAPI:
    """Build the application (injectable for tests)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = database
    app.state.model_service = model_service

    app.add_middleware(RequestInstrumentationMiddleware)
    for router in ROUTERS:
        app.include_router(router)
    app.include_router(graphql_router, prefix="/graphql")

    _register_state_collector(settings)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.exception_handler(DataValidationError)
    async def validation_error_handler(request: Request, exc: DataValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": str(exc),
                "failures": exc.failures,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(ModelNotAvailableError)
    async def model_unavailable_handler(
        request: Request, exc: ModelNotAvailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc), "request_id": _request_id(request)},
        )

    @app.exception_handler(MLOpsError)
    async def domain_error_handler(request: Request, exc: MLOpsError) -> JSONResponse:
        logger.error("domain error", extra={"event": "domain_error"}, exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "request_id": _request_id(request)},
        )

    return app


app = create_app()
