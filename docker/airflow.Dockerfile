# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Airflow image with the MLOps package installed.
# Build:  docker build -f docker/airflow.Dockerfile -t interest-rate-airflow:local .
#
# The base image already provides Airflow and its pinned dependencies; the
# project requirements add the ML stack (mlflow, xgboost, evidently, dvc, ...).
# ---------------------------------------------------------------------------
FROM apache/airflow:2.10.4-python3.12

USER airflow

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLOPS_PROJECT_ROOT=/opt/airflow/project

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Project code: editable install keeps PROJECT_ROOT pointing at the mounted repo.
COPY --chown=airflow:root pyproject.toml README.md /opt/airflow/project/
COPY --chown=airflow:root src/ /opt/airflow/project/src/
COPY --chown=airflow:root api/ /opt/airflow/project/api/
COPY --chown=airflow:root configs/ /opt/airflow/project/configs/
COPY --chown=airflow:root scripts/ /opt/airflow/project/scripts/
COPY --chown=airflow:root alembic/ /opt/airflow/project/alembic/
COPY --chown=airflow:root alembic.ini /opt/airflow/project/

RUN pip install --no-cache-dir --no-deps -e /opt/airflow/project

ENV PYTHONPATH=/opt/airflow/project/src:/opt/airflow/project
