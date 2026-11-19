"""DAG 1 - data ingestion and validation.

Manual by default because the Kaggle dataset is static: new observations do not
arrive on their own. Set ``INGESTION_SCHEDULE`` (e.g. ``@weekly``) once a live
source such as FRED is configured, and ``INGESTION_FETCH=true`` to re-download.
"""

from __future__ import annotations

from airflow.decorators import dag, task
from dag_utils import DEFAULT_ARGS, START_DATE, fetch_enabled, jsonable, schedule

from interest_rate_mlops.pipelines import run_ingestion


@dag(
    dag_id="ingestion_pipeline",
    description="Ingest, validate and version the interest-rate dataset",
    schedule=schedule("INGESTION_SCHEDULE"),
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "ingestion"],
    doc_md=__doc__,
)
def ingestion_pipeline() -> None:
    @task(task_id="ingest_and_validate")
    def ingest() -> dict:
        outcome = run_ingestion(fetch=fetch_enabled())
        result = jsonable(outcome.as_dict())
        if not outcome.validation_passed:
            raise ValueError("raw data validation failed; see reports/data_quality")
        return result

    @task(task_id="check_quality")
    def check_quality(outcome: dict) -> dict:
        rows_in, rows_out = outcome.get("rows_in"), outcome.get("rows_out")
        if not rows_out:
            raise ValueError("no monthly observations were produced")
        if rows_in and rows_out and rows_out / rows_in < 0.5:
            # Intra-month policy rows are expected to be dropped, but losing more
            # than half the rows means the schema or frequency changed.
            raise ValueError(f"suspicious row reduction: {rows_in} -> {rows_out}")
        return outcome

    @task(task_id="record_lineage")
    def record_lineage(outcome: dict) -> None:
        print(
            "ingestion complete",
            {
                "dataset_version": outcome.get("dataset_version"),
                "rows_out": outcome.get("rows_out"),
                "dvc": outcome.get("dvc"),
                "panel_path": outcome.get("panel_path"),
            },
        )

    record_lineage(check_quality(ingest()))


ingestion_pipeline()
