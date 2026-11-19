"""Model comparison report.

The report is deliberately machine-readable first (JSON/CSV) and human-readable
second (Markdown), so it can be attached to a PR, stored in the MLflow run,
indexed in PostgreSQL and rendered in the README without reformatting.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from interest_rate_mlops.config import SelectionConfig
from interest_rate_mlops.logging_utils import get_logger

if TYPE_CHECKING:  # avoids an evaluation <-> training import cycle at runtime
    from interest_rate_mlops.training.splits import Split

logger = get_logger(__name__)

REPORT_COLUMNS = [
    "model_name",
    "model_type",
    "test_mae",
    "test_rmse",
    "test_mape",
    "test_mape_coverage_pct",
    "cv_rmse_mean",
    "cv_mae_mean",
    "cv_rmse_std",
    "train_duration_seconds",
    "n_train",
    "n_test",
]


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_comparison(
    models: list[dict[str, Any]],
    *,
    dataset_name: str,
    dataset_version: str,
    target_name: str,
    target_unit: str,
    horizon: int,
    split: Split,
    feature_metadata: dict[str, Any],
    selection: SelectionConfig,
    selected_model: str,
    evaluation_date: str,
    cv_splits: int,
    dvc_hash: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Create the comparison payload shared by every consumer."""
    ranked = sorted(
        models,
        key=lambda item: (
            item.get("cv_rmse_mean") if item.get("cv_rmse_mean") is not None else float("inf"),
            item.get("cv_mae_mean") if item.get("cv_mae_mean") is not None else float("inf"),
            item["model_name"],
        ),
    )
    return {
        "report_version": 1,
        "evaluation_date": evaluation_date,
        "dataset": {
            "name": dataset_name,
            "version": dataset_version,
            "dvc_hash": dvc_hash,
            "git_commit": git_commit,
        },
        "target": {
            "name": target_name,
            "unit": target_unit,
            "horizon_observations": horizon,
            "frequency": "monthly",
        },
        "features": feature_metadata,
        "split": split.as_dict(),
        "validation": {
            "strategy": "expanding_window",
            "n_splits": cv_splits,
            "held_out_from_selection": True,
        },
        "selection": {
            "primary_metric": selection.primary_metric,
            "secondary_metric": selection.secondary_metric,
            "minimize": selection.minimize,
            "selected_model": selected_model,
            "rule": (
                "lowest mean walk-forward CV RMSE on the training block; the held-out "
                "test block is reported for every model but never used to pick the winner"
            ),
        },
        "ranking": [item["model_name"] for item in ranked],
        "models": [{key: _round(value) for key, value in item.items()} for item in models],
        "notes": [
            "MAE and RMSE are in percentage points and are the primary metrics.",
            "MAPE excludes observations with |actual| < 0.5 pp; coverage is reported.",
            "The test block is the final chronological 20% and is never used for fitting.",
        ],
    }


def _markdown(comparison: dict[str, Any]) -> str:
    header = (
        "| Model | MAE | RMSE | MAPE (%) | MAPE coverage (%) | CV RMSE | CV MAE | "
        "Train (s) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for model in comparison["models"]:
        rows.append(
            "| {name} | {mae} | {rmse} | {mape} | {cov} | {cv_rmse} | {cv_mae} | {train} |".format(
                name=model["model_name"],
                mae=_fmt(model.get("test_mae")),
                rmse=_fmt(model.get("test_rmse")),
                mape=_fmt(model.get("test_mape"), 2),
                cov=_fmt(model.get("test_mape_coverage_pct"), 1),
                cv_rmse=_fmt(model.get("cv_rmse_mean")),
                cv_mae=_fmt(model.get("cv_mae_mean")),
                train=_fmt(model.get("train_duration_seconds"), 3),
            )
        )

    split = comparison["split"]
    return "\n".join(
        [
            "# Model comparison report",
            "",
            f"- Evaluation date: `{comparison['evaluation_date']}`",
            f"- Dataset: `{comparison['dataset']['name']}` version "
            f"`{comparison['dataset']['version']}`",
            f"- Target: `{comparison['target']['name']}` "
            f"({comparison['target']['unit']}), horizon "
            f"{comparison['target']['horizon_observations']} observation(s)",
            f"- Split: chronological, train {split['train_start']}..{split['train_end']} "
            f"({split['n_train']} rows), test {split['test_start']}..{split['test_end']} "
            f"({split['n_test']} rows)",
            f"- Selection rule: {comparison['selection']['rule']}",
            f"- Selected model: **{comparison['selection']['selected_model']}**",
            "",
            "## Results",
            "",
            header + "\n".join(rows),
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in comparison["notes"]],
            "",
        ]
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def save_comparison(comparison: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, CSV and Markdown variants of the comparison."""
    from interest_rate_mlops.storage import write_json

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = write_json(directory / "model_comparison.json", comparison)

    frame = pd.DataFrame(comparison["models"])
    present = [column for column in REPORT_COLUMNS if column in frame.columns]
    csv_path = directory / "model_comparison.csv"
    frame[present].to_csv(csv_path, index=False)

    md_path = directory / "model_comparison.md"
    md_path.write_text(_markdown(comparison), encoding="utf-8")

    logger.info(
        "saved model comparison report",
        extra={"event": "comparison_saved", "path": str(json_path)},
    )
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def utc_timestamp() -> str:
    """Compact UTC timestamp used for artifact directory names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
