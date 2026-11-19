"""Orchestration entry points shared by Airflow DAGs and the CLI scripts.

Each pipeline is a plain function with typed inputs/outputs, so it can be unit
tested, called from a DAG task, or run by hand without Airflow.
"""

from interest_rate_mlops.pipelines.ingestion_pipeline import IngestionOutcome, run_ingestion
from interest_rate_mlops.pipelines.monitoring_pipeline import (
    MonitoringOutcome,
    run_monitoring,
)
from interest_rate_mlops.pipelines.promotion_pipeline import PromotionOutcome, run_promotion
from interest_rate_mlops.pipelines.training_pipeline import TrainingOutcome, run_training

__all__ = [
    "IngestionOutcome",
    "MonitoringOutcome",
    "PromotionOutcome",
    "TrainingOutcome",
    "run_ingestion",
    "run_monitoring",
    "run_promotion",
    "run_training",
]
