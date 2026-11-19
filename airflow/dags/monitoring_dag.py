"""DAG 4 - data drift and model performance monitoring.

Runs Evidently reports, evaluates data drift and (when actuals exist) model
performance as two independent signals, stores the artifacts and requests
retraining only when a configured threshold is crossed.
"""

from __future__ import annotations

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from dag_utils import DEFAULT_ARGS, START_DATE, auto_retrain_enabled, jsonable, schedule

from interest_rate_mlops.pipelines import run_monitoring


@dag(
    dag_id="monitoring_pipeline",
    description="Evidently drift and performance monitoring with retraining trigger",
    schedule=schedule("MONITORING_SCHEDULE"),
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "monitoring"],
    doc_md=__doc__,
)
def monitoring_pipeline() -> None:
    @task(task_id="run_monitoring")
    def monitor() -> dict:
        return jsonable(run_monitoring().as_dict())

    @task(task_id="log_status")
    def log_status(outcome: dict) -> dict:
        print("drift status:", outcome.get("status"))
        print("data drift detected:", outcome.get("data_drift"))
        print("performance status:", outcome.get("performance_status"))
        print("retrain recommended:", outcome.get("retrain_recommended"))
        print("report:", outcome.get("report_path"))
        if outcome.get("status") == "critical":
            # Surfaces in the task log and can drive an Airflow alerting rule.
            print("ALERT: drift status critical")
        return outcome

    @task.short_circuit(task_id="should_retrain")
    def should_retrain(outcome: dict) -> bool:
        return bool(outcome.get("retrain_recommended")) and auto_retrain_enabled()

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    should_retrain(log_status(monitor())) >> trigger_training


monitoring_pipeline()
