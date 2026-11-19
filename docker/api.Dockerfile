# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# API / MLflow image (Python 3.12, CPU only).
# Build:  docker build -f docker/api.Dockerfile -t interest-rate-api:local .
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

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

# Dependencies first so code changes do not invalidate the layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code and configuration.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir --no-deps -e .

# Non-root runtime user.
RUN useradd --create-home --uid 1000 mlops \
    && mkdir -p /app/data /app/reports /app/mlruns \
    && chown -R mlops:mlops /app
USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
