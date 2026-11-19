"""Pandera schema for the modelling panel.

The raw file is validated structurally in :mod:`validators`; this schema guards
the *contract* every downstream stage relies on: a unique month-start index, a
strict column set, and a target inside a plausible range.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.errors import SchemaError, SchemaErrors

from interest_rate_mlops.config import DatasetConfig
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.validation.validators import CheckResult, ValidationReport

logger = get_logger(__name__)


def _is_month_start(index: pd.Index) -> bool:
    """True when every index entry is the first day of its month.

    Pandera hands the check a plain ``Series``, which has no ``.dt`` accessor for
    datetime values, so the values are normalised through ``DatetimeIndex``.
    """
    values = pd.DatetimeIndex(index)
    return bool((values.day == 1).all())


def panel_schema(cfg: DatasetConfig) -> pa.DataFrameSchema:
    """Build the panel schema for the configured target and extra series."""
    low, high = cfg.validation.allowed_target_range
    columns: dict[str, pa.Column] = {
        cfg.target.slug: pa.Column(
            float,
            nullable=False,
            checks=pa.Check.in_range(low, high, include_min=True),
            description=f"{cfg.target.column} in {cfg.target.unit}",
        )
    }
    for series in cfg.exogenous:
        columns[series.slug] = pa.Column(
            float,
            nullable=True,
            required=True,
            description=f"{series.column} ({series.frequency})",
        )

    month_start = pa.Check(
        _is_month_start,
        error="panel index must contain month-start dates only",
    )
    return pa.DataFrameSchema(
        columns=columns,
        index=pa.Index(pa.DateTime, name="date", unique=True, checks=month_start),
        strict=True,
        coerce=True,
        checks=pa.Check(lambda frame: len(frame) > 0, error="panel must not be empty"),
    )


def validate_panel(panel: pd.DataFrame, cfg: DatasetConfig) -> ValidationReport:
    """Validate the monthly panel and return a structured report."""
    report = ValidationReport(dataset=f"{cfg.name}:panel", n_rows=int(len(panel)))
    try:
        panel_schema(cfg).validate(panel, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        report.add(
            CheckResult(
                name="panel_schema",
                passed=False,
                detail=_describe_schema_error(exc),
            )
        )
        return report

    report.add(
        CheckResult(
            name="panel_schema",
            passed=True,
            detail=(
                f"{len(panel)} monthly rows, "
                f"{panel.index.min().date()}..{panel.index.max().date()}, "
                f"columns={list(panel.columns)}"
            ),
        )
    )
    report.add(
        CheckResult(
            name="target_completeness",
            passed=bool(panel[cfg.target.slug].notna().all()),
            detail=f"{int(panel[cfg.target.slug].isna().sum())} missing target observations",
        )
    )
    logger.info(
        "panel validation finished",
        extra={"event": "panel_validated", "dataset": cfg.name},
    )
    return report


def _describe_schema_error(error: Exception) -> str:
    """Condense a Pandera error into one actionable line."""
    failure_cases = getattr(error, "failure_cases", None)
    if failure_cases is not None and len(failure_cases):
        try:
            details = [
                f"{row.get('column') or 'index'}: {row.get('check')} -> {row.get('error')}"
                for row in failure_cases.head(5).to_dict("records")
            ]
            return "; ".join(details)
        except AttributeError:  # pragma: no cover - defensive
            pass
    return f"{error.__class__.__name__}: {error}"[:1000]
