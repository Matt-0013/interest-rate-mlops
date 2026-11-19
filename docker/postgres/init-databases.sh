#!/bin/bash
# ---------------------------------------------------------------------------
# Creates the two auxiliary databases on first container start.
#
# The three databases stay logically separate even though one PostgreSQL server
# hosts them:
#   * $POSTGRES_DB  - application database (predictions, training metadata)
#   * mlflow        - MLflow tracking server + model registry
#   * airflow       - Airflow metadata
# ---------------------------------------------------------------------------
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE mlflow OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
    SELECT 'CREATE DATABASE airflow OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
EOSQL

echo "databases ready: $POSTGRES_DB, mlflow, airflow"
