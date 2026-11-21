# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# API / MLflow image (Python 3.12, CPU only).
# Build:  docker build -f docker/api.Dockerfile -t interest-rate-api:local .
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Build-only pip resilience. These are ARGs rather than ENV on purpose: Docker
# exposes them to the RUN steps below but never writes them into the runtime
# image environment. Override without editing this file, e.g.
#   docker build --build-arg PIP_INSTALL_ATTEMPTS=8 ...
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=15
ARG PIP_INSTALL_ATTEMPTS=5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLOPS_PROJECT_ROOT=/app

WORKDIR /app

# curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code changes do not invalidate the layer. A transient
# PyPI timeout on a slow link retries instead of failing the whole build.
COPY requirements.txt ./
RUN set -eu; \
    for attempt in $(seq 1 "$PIP_INSTALL_ATTEMPTS"); do \
        echo "pip install attempt ${attempt}/${PIP_INSTALL_ATTEMPTS} (retries=${PIP_RETRIES}, timeout=${PIP_DEFAULT_TIMEOUT}s)"; \
        if pip install --no-cache-dir -r requirements.txt; then \
            exit 0; \
        fi; \
        echo "pip install failed on attempt ${attempt}; retrying in 20s" >&2; \
        sleep 20; \
    done; \
    echo "pip install failed after ${PIP_INSTALL_ATTEMPTS} attempts" >&2; \
    exit 1

# Application code and configuration.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

RUN set -eu; \
    for attempt in $(seq 1 "$PIP_INSTALL_ATTEMPTS"); do \
        if pip install --no-cache-dir --no-deps -e .; then \
            exit 0; \
        fi; \
        echo "editable install failed on attempt ${attempt}; retrying in 20s" >&2; \
        sleep 20; \
    done; \
    exit 1

# Non-root runtime user.
RUN useradd --create-home --uid 1000 mlops \
    && mkdir -p /app/data /app/reports /app/mlruns \
    && chown -R mlops:mlops /app
USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
