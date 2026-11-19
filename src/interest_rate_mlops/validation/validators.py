"""Raw-dataset validation.

Checks are *reported* individually so a failing pipeline explains exactly which
assumption broke. Hard failures raise :class:`DataValidationError`; soft findings
(high missingness in a sparse series, statistical outliers) are recorded but do
not block training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from interest_rate_mlops.config import DatasetConfig
from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.ingestion.loader import build_dates, coerce_numeric_series
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single named check."""

    name: str
    passed: bool
    detail: str
    severity: str = "error"  # error | warning | info

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierReport:
    """IQR-based outlier summary for one numeric series."""

    column: str
    n_outliers: int
    lower_bound: float
    upper_bound: float
    samples: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated validation outcome."""

    dataset: str
    n_rows: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no *error*-severity check failed."""
        return all(check.passed for check in self.checks if check.severity == "error")

    @property
    def failures(self) -> list[str]:
        return [
            check.name for check in self.checks if not check.passed and check.severity == "error"
        ]

    def add(self, check: CheckResult) -> ValidationReport:
        self.checks.append(check)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_rows": self.n_rows,
            "passed": self.passed,
            "failures": self.failures,
            "checks": [check.as_dict() for check in self.checks],
        }

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise DataValidationError(
                f"dataset {self.dataset!r} failed validation: {', '.join(self.failures)}",
                failures=self.failures,
            )


def detect_outliers(series: pd.Series, column: str, iqr_factor: float = 3.0) -> OutlierReport:
    """Flag values outside ``median +/- iqr_factor * IQR``.

    Extreme interest-rate observations are historically real (e.g. the 1981
    fed funds peak), so outliers are reported for review, never removed here.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return OutlierReport(column=column, n_outliers=0, lower_bound=np.nan, upper_bound=np.nan)

    q1, q3 = float(numeric.quantile(0.25)), float(numeric.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    mask = (numeric < lower) | (numeric > upper)
    return OutlierReport(
        column=column,
        n_outliers=int(mask.sum()),
        lower_bound=round(lower, 4),
        upper_bound=round(upper, 4),
        samples=[f"{value:g}" for value in numeric[mask].head(5).tolist()],
    )


def validate_raw_dataset(
    frame: pd.DataFrame, cfg: DatasetConfig
) -> tuple[ValidationReport, list[OutlierReport]]:
    """Validate the raw CSV before any transformation.

    Returns:
        The report plus the per-series outlier summaries (for the data-quality
        artifact).
    """
    report = ValidationReport(dataset=cfg.name, n_rows=int(len(frame)))
    expected = list(cfg.validation.required_columns)
    actual = list(frame.columns)

    missing = [column for column in expected if column not in actual]
    report.add(
        CheckResult(
            name="required_columns",
            passed=not missing,
            detail="all required columns present" if not missing else f"missing: {missing}",
        )
    )
    unexpected = [column for column in actual if column not in expected]
    report.add(
        CheckResult(
            name="unexpected_columns",
            passed=not unexpected,
            detail=(
                "no unexpected columns"
                if not unexpected
                else f"unexpected columns present (schema drift): {unexpected}"
            ),
        )
    )
    if missing:
        # Downstream checks depend on these columns, so stop here and let the
        # caller decide whether to raise *after* persisting the report.
        return report, []

    # --- dates -------------------------------------------------------------
    try:
        dates = build_dates(frame, cfg.date_columns)
        date_check = CheckResult(
            name="valid_dates",
            passed=True,
            detail=f"{len(dates)} dates parsed, {dates.min().date()}..{dates.max().date()}",
        )
    except DataValidationError as exc:
        date_check = CheckResult(name="valid_dates", passed=False, detail=str(exc))
    report.add(date_check)
    if not date_check.passed:
        return report, []

    # --- duplicates --------------------------------------------------------
    duplicate_rows = int(frame.duplicated(subset=cfg.validation.duplicate_key).sum())
    report.add(
        CheckResult(
            name="duplicate_records",
            passed=duplicate_rows == 0,
            detail=(
                "no duplicate records"
                if duplicate_rows == 0
                else f"{duplicate_rows} duplicate rows on {cfg.validation.duplicate_key}"
            ),
        )
    )

    # --- numeric coercion --------------------------------------------------
    bad_values: dict[str, list[str]] = {}
    for column in frame.columns:
        if column in cfg.date_columns:
            continue
        _, unparseable = coerce_numeric_series(frame[column])
        if unparseable.any():
            bad_values[column] = sorted(
                frame.loc[unparseable, column].astype(str).unique().tolist()
            )[:5]
    report.add(
        CheckResult(
            name="numeric_values",
            passed=not bad_values,
            detail="all series are numeric" if not bad_values else f"non-numeric: {bad_values}",
        )
    )

    # --- missingness -------------------------------------------------------
    # The raw file interleaves scheduled month-start observations with
    # intra-month policy-decision rows that legitimately carry no effective rate.
    # Missingness is therefore measured on scheduled rows only; measuring it on
    # every row would flag a structural property as a data problem.
    target, _ = coerce_numeric_series(frame[cfg.target.column])
    scheduled = dates.dt.day == 1
    scheduled_target = target[scheduled]
    target_missing = float(scheduled_target.isna().mean()) if len(scheduled_target) else 1.0
    raw_missing = float(target.isna().mean())
    report.add(
        CheckResult(
            name="target_missingness",
            passed=target_missing <= cfg.validation.max_missing_fraction_target,
            detail=(
                f"{target_missing:.2%} of scheduled (month-start) {cfg.target.column!r} "
                f"observations are missing (limit "
                f"{cfg.validation.max_missing_fraction_target:.0%}); "
                f"{raw_missing:.2%} of all raw rows lack it, which includes the "
                f"{int((~scheduled).sum())} intra-month rows"
            ),
        )
    )
    for column in frame.columns:
        if column in cfg.date_columns:
            continue
        missing_fraction = float(frame[column].isna().mean())
        report.add(
            CheckResult(
                name=f"missingness::{column}",
                passed=True,  # informational: sparse macro series are expected
                severity="info",
                detail=f"{missing_fraction:.2%} missing "
                f"(limit {cfg.validation.max_missing_fraction:.0%})",
            )
        )

    # --- target plausibility ----------------------------------------------
    target_numeric = target
    low, high = cfg.validation.allowed_target_range
    observed = target_numeric.dropna()
    within = observed.between(low, high).all() if not observed.empty else True
    report.add(
        CheckResult(
            name="target_range",
            passed=bool(within),
            detail=(
                f"{len(observed)} observations within [{low}, {high}]"
                if within
                else f"values outside [{low}, {high}]: "
                f"{observed[~observed.between(low, high)].head(5).tolist()}"
            ),
        )
    )

    # --- outliers (informational) -----------------------------------------
    outliers = [
        detect_outliers(frame[column], column, cfg.validation.outlier_iqr_factor)
        for column in frame.columns
        if column not in cfg.date_columns
    ]
    flagged = {item.column: item.n_outliers for item in outliers if item.n_outliers}
    report.add(
        CheckResult(
            name="outliers_iqr",
            passed=True,
            severity="info",
            detail=f"flagged for review (not removed): {flagged}" if flagged else "none flagged",
        )
    )

    logger.info(
        "raw dataset validation finished",
        extra={
            "event": "validation_complete",
            "dataset": cfg.name,
            "status_code": 200 if report.passed else 422,
        },
    )
    return report, outliers
