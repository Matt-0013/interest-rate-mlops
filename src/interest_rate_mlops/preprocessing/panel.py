"""Monthly panel construction.

The raw file is not a tidy monthly table: it interleaves monthly observations
(published on the 1st) with 151 intra-month policy-decision rows that only carry
a target rate. Two rules make the panel trustworthy:

1. only scheduled observations (month-start rows) become panel rows, so an
   intra-month decision can never masquerade as a month-end observation;
2. the panel is re-indexed onto the *declared* monthly calendar, so a month with
   no observation is visibly missing instead of silently absent. Such interior
   gaps are a hard error for the target - the pipeline refuses to invent values.

Forward-filling is only ever applied to explicitly configured sparse exogenous
series, is recorded in the report, and is never applied to the target.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from interest_rate_mlops.config import DatasetConfig
from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.ingestion.loader import (
    build_dates,
    coerce_numeric_series,
    infer_frequency,
)
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.storage import read_json, read_parquet, write_json, write_parquet

logger = get_logger(__name__)


@dataclass(frozen=True)
class PanelReport:
    """What happened while building the panel (persisted with the artifact)."""

    target: str
    anchor: str
    rows_in: int
    rows_out: int
    frequency: str | None
    intra_period_rows_dropped: int
    intra_period_dates_sample: list[str] = field(default_factory=list)
    intra_period_columns: dict[str, int] = field(default_factory=dict)
    missing_months: list[str] = field(default_factory=list)
    target_gaps: list[str] = field(default_factory=list)
    trimmed_leading: int = 0
    trimmed_trailing: int = 0
    asof_fills: dict[str, int] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    start: str | None = None
    end: str | None = None

    @property
    def has_target_gaps(self) -> bool:
        return bool(self.target_gaps)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PanelBuildResult:
    """Panel plus its build report."""

    panel: pd.DataFrame
    report: PanelReport

    @property
    def target_column(self) -> str:
        return self.report.target


def _coerce_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a published column to float, failing loudly on unparseable text."""
    parsed, unparseable = coerce_numeric_series(frame[column])
    if unparseable.any():
        samples = sorted(frame.loc[unparseable, column].astype(str).unique().tolist())[:5]
        raise DataValidationError(
            f"column {column!r} contains non-numeric values: {samples}",
            failures=["numeric_values"],
        )
    return parsed


def build_monthly_panel(raw: pd.DataFrame, cfg: DatasetConfig) -> PanelBuildResult:
    """Turn the raw snapshot into a validated monthly panel.

    Args:
        raw: Raw CSV frame (string dtypes, published column names).
        cfg: Dataset configuration (target, extra series, panel rules).

    Returns:
        The panel indexed by month-start with canonical column slugs.

    Raises:
        DataValidationError: on invalid dates, multiple observations for the
            same month, interior target gaps, or an empty result.
    """
    slug_map = cfg.series_slugs()
    missing_series = [column for column in slug_map if column not in raw.columns]
    if missing_series:
        raise DataValidationError(
            f"configured series not present in raw data: {missing_series}",
            failures=["required_columns"],
        )

    dates = build_dates(raw, cfg.date_columns)
    frame = raw.assign(date=dates)

    # Build from the original row index first, then swap in the observation date:
    # constructing the frame *with* a DatetimeIndex would align the integer-indexed
    # series and silently produce an all-NaN panel.
    series = pd.DataFrame(
        {slug: _coerce_series(frame, column) for column, slug in slug_map.items()}
    )
    series.index = pd.DatetimeIndex(frame["date"].to_numpy(), name="date")
    series = series.sort_index()

    # --- separate scheduled observations from intra-period events ----------
    is_anchor = series.index.day == 1
    intra = series.loc[~is_anchor]
    intra_columns = (
        {column: int(count) for column, count in intra.notna().sum().items() if count}
        if not intra.empty
        else {}
    )
    if not intra.empty:
        if not cfg.panel.drop_intra_period_rows:
            raise DataValidationError(
                f"{len(intra)} intra-month rows present and "
                "panel.drop_intra_period_rows is false; multiple observations per "
                "month cannot be placed on a monthly calendar",
                failures=["intra_period_rows"],
            )
        logger.info(
            "dropping intra-month rows from the monthly panel",
            extra={"event": "intra_period_rows_dropped", "status_code": 200},
        )

    monthly = series.loc[is_anchor]
    duplicates = monthly.index[monthly.index.duplicated()].unique()
    if len(duplicates):
        raise DataValidationError(
            f"multiple observations for the same month: "
            f"{[str(value.date()) for value in duplicates[:5]]}",
            failures=["duplicate_records"],
        )

    if monthly.empty:
        raise DataValidationError("no month-start observations found", failures=["empty_panel"])

    # --- re-index onto the declared calendar -------------------------------
    calendar = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    panel = monthly.reindex(calendar)
    panel.index.name = "date"

    target = cfg.target.slug
    observed = panel[target].notna()
    if not observed.any():
        raise DataValidationError(
            f"target {cfg.target.column!r} has no observations in the panel",
            failures=["empty_target"],
        )

    first_valid = panel.index[observed][0]
    last_valid = panel.index[observed][-1]
    interior = panel.loc[first_valid:last_valid, target]
    target_gaps = [str(date.date()) for date in interior.index[interior.isna()]]
    fully_empty_months = panel.index[panel.isna().all(axis=1)]
    missing_months = [str(value.date()) for value in fully_empty_months]

    if target_gaps and cfg.panel.require_contiguous_target:
        raise DataValidationError(
            f"target has {len(target_gaps)} interior missing months "
            f"(first: {target_gaps[:5]}); the panel refuses to impute the target",
            failures=["target_gaps"],
        )

    trimmed_leading = int((panel.index < first_valid).sum())
    trimmed_trailing = int((panel.index > last_valid).sum())
    panel = panel.loc[first_valid:last_valid]

    # --- as-of carry-forward for explicitly configured sparse series -------
    asof_fills: dict[str, int] = {}
    for series_cfg in cfg.exogenous:
        if series_cfg.asof_fill != "last_known" or series_cfg.slug not in panel.columns:
            continue
        before_missing = panel[series_cfg.slug].isna()
        panel[series_cfg.slug] = panel[series_cfg.slug].ffill()
        filled = int((before_missing & panel[series_cfg.slug].notna()).sum())
        asof_fills[series_cfg.slug] = filled

    # --- apply the configured window --------------------------------------
    if cfg.panel.start is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(cfg.panel.start)]
    if cfg.panel.end is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(cfg.panel.end)]
    if panel.empty:
        raise DataValidationError(
            "panel is empty after applying panel.start/panel.end",
            failures=["empty_panel"],
        )
    # When contiguity is explicitly waived the gap stays visible in the panel (and
    # in the report); feature engineering then refuses the NaN target rather than
    # imputing it.
    if cfg.panel.require_contiguous_target and panel[target].isna().any():
        raise DataValidationError(
            "configured window includes months without a target observation",
            failures=["target_gaps"],
        )

    report = PanelReport(
        target=target,
        anchor=cfg.panel.anchor,
        rows_in=int(len(raw)),
        rows_out=int(len(panel)),
        frequency=infer_frequency(panel.index.to_series()) or cfg.target.frequency,
        intra_period_rows_dropped=int(len(intra)),
        intra_period_dates_sample=[str(value.date()) for value in intra.index[:10]],
        intra_period_columns=intra_columns,
        missing_months=missing_months,
        target_gaps=target_gaps,
        trimmed_leading=trimmed_leading,
        trimmed_trailing=trimmed_trailing,
        asof_fills=asof_fills,
        columns=list(panel.columns),
        start=str(panel.index.min().date()),
        end=str(panel.index.max().date()),
    )
    logger.info(
        "built monthly panel",
        extra={
            "event": "panel_built",
            "rows_in": int(len(raw)),
            "rows_out": int(len(panel)),
            "start": report.start,
            "end": report.end,
        },
    )
    return PanelBuildResult(panel=panel, report=report)


def save_panel(panel: pd.DataFrame, path: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist the panel as parquet plus a JSON sidecar describing its build."""
    resolved = write_parquet(panel.reset_index().set_index("date"), path)
    write_json(Path(f"{resolved}.meta.json"), metadata)
    return resolved


def load_panel(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a persisted panel and its metadata sidecar."""
    resolved = Path(path)
    frame = read_parquet(resolved)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    meta_path = Path(f"{resolved}.meta.json")
    metadata = read_json(meta_path, default={})
    if not isinstance(metadata, dict):  # pragma: no cover - defensive
        metadata = json.loads(json.dumps(metadata))
    return frame.sort_index(), metadata
