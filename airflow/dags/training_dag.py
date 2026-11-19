"""DAG 2 - model training and experiment tracking.

Trains all three candidates on the same split, logs every run to MLflow and
registers the selected one under the ``challenger`` alias, then triggers the
promotion DAG. Training never moves the ``champion`` alias itself.
"""

from __future__ import annotations

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from dag_utils import DEFAULT_ARGS, START_DATE, jsonable, schedule

from interest_rate_mlops.pipelines import run_training


@dag(
    dag_id="training_pipeline",
    description="Train, evaluate and register the three candidate models",
    schedule=schedule("TRAINING_SCHEDULE"),
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training"],
    doc_md=__doc__,
)
def training_pipeline() -> None:
    @task(task_id="train_and_log")
    def train() -> dict:
        return jsonable(run_training().as_dict())

    @task(task_id="summarise_comparison")
    def summarise(outcome: dict) -> dict:
        if not outcome.get("candidate_version"):
            raise ValueError("no candidate was registered; check the MLflow backend")
        scores = outcome.get("scores", {})
        print("model comparison", scores)
        print("selected model:", outcome.get("best_model"))
        print("candidate version:", outcome.get("candidate_version"))
        print("comparison report:", outcome.get("artifacts_dir"))
        return outcome

    trigger_promotion = TriggerDagRunOperator(
        task_id="trigger_promotion",
        trigger_dag_id="promotion_pipeline",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=30,
    )

    summarise(train()) >> trigger_promotion


training_pipeline()
