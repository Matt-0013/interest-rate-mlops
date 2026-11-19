"""Request instrumentation.

Every request gets a correlation id, a structured log line and Prometheus
metrics. Endpoint labels use the FastAPI route template (not the raw path) so
cardinality stays bounded.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api import metrics
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_METRICS_PATH = "/metrics"


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or "unresolved"


class RequestInstrumentationMiddleware(BaseHTTPMiddleware):
    """Correlation id + latency + status metrics for every request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000.0
            endpoint = _endpoint_label(request)
            metrics.REQUEST_COUNT.labels(request.method, endpoint, "500").inc()
            metrics.HTTP_ERRORS.labels(endpoint, "500").inc()
            metrics.REQUEST_LATENCY.labels(endpoint).observe(duration_ms / 1000.0)
            logger.exception(
                "unhandled request error",
                extra={
                    "event": "request_failed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000.0
        endpoint = _endpoint_label(request)
        status = str(response.status_code)

        if request.url.path != _METRICS_PATH:
            metrics.REQUEST_COUNT.labels(request.method, endpoint, status).inc()
            metrics.REQUEST_LATENCY.labels(endpoint).observe(duration_ms / 1000.0)
            if response.status_code >= 400:
                metrics.HTTP_ERRORS.labels(endpoint, status).inc()

        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request handled",
            extra={
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
