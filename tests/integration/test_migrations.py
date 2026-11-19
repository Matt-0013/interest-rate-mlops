"""Alembic migrations apply cleanly and are reversible."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from interest_rate_mlops.config import PROJECT_ROOT

EXPECTED_TABLES = {
    "training_runs",
    "model_metadata",
    "model_deployment_history",
    "prediction_records",
    "data_ingestion_logs",
    "monitoring_runs",
    "pipeline_executions",
}


def test_migrations_upgrade_and_downgrade(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'migrations.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    assert tables >= EXPECTED_TABLES
    assert "alembic_version" in tables

    # Columns the API depends on must exist after the migration.
    prediction_columns = {
        column["name"] for column in inspect(engine).get_columns("prediction_records")
    }
    assert {
        "request_id",
        "model_version",
        "prediction",
        "features",
        "created_at",
    } <= prediction_columns

    command.downgrade(config, "base")
    remaining = set(inspect(engine).get_table_names())
    assert not (EXPECTED_TABLES & remaining)
    engine.dispose()
