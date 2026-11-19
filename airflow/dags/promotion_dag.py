"""DAG 3 - model evaluation and promotion.

Applies every configured gate before moving the ``champion`` alias. A rejection
is a normal, recorded outcome: the incumbent stays in place and the reason is
written to reports/promotion and to the database.
"""

from __future__ import annotations

from airflow.decorators import dag, task
from dag_utils import DEFAULT_ARGS, START_DATE, jsonable

from interest_rate_mlops.pipelines import run_promotion


@dag(
    dag_id="promotion_pipeline",
    description="Gate and promote the registered candidate to champion",
    schedule=None,  # triggered by the training DAG or run manually
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "promotion"],
    doc_md=__doc__,
)
def promotion_pipeline() -> None:
    @task(task_id="evaluate_candidate")
    def evaluate() -> dict:
        return jsonable(run_promotion().as_dict())

    @task(task_id="report_decision")
    def report(outcome: dict) -> None:
        decision = outcome["decision"]
        print("candidate version:", outcome.get("candidate_version"))
        print("promoted:", outcome.get("promoted"))
        for gate in decision.get("gates", []):
            print(f"gate {gate['name']}: {'PASS' if gate['passed'] else 'FAIL'} - {gate['detail']}")
        print("summary:", decision.get("summary"))
        if not outcome.get("promoted") and not decision.get("gates"):
            raise ValueError("promotion produced neither a decision nor gates; check the registry")

    report(evaluate())


promotion_pipeline()
