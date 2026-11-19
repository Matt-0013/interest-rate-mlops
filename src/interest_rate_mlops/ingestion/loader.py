"""Raw dataset loading and structural profiling.

The raw dataset stores its date as three integer columns (``Year``, ``Month``,
``Day``) and mixes two row types: monthly observations published on the first of
the month and intra-month policy-decision rows that only carry a target rate.
Both facts were verified against the actual file (see ``docs/dataset.md``); the
loader therefore never assumes a layout, it derives one and reports on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from interest_rate_mlops.config import DatasetConfig
from interest_rate_mlops.exceptions import DataValidationError, IngestionError
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)

FREQUENCY_LABELS: tuple[tuple[float, str], ...] = (
    (2.5, "daily"),
    (10.0, "weekly"),
    (45.0, "monthly"),
    (120.0, "quarterly"),
)


@dataclass(frozen=True)
class ColumnProfile:
    """Per-column structural summary."""

    name: str
    non_null: int
    null_fraction: float
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    distinct_values: int | None = None
    first_observation: str | None = None
    non_numeric_samples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RawDatasetProfile:
    """Everything the pipeline needs to know about the raw file."""

    path: str
    n_rows: int
    n_columns: int
    columns: list[str]
    date_range: tuple[str, str] | None
    n_unique_dates: int
    duplicate_dates: list[str]
    n_intra_period_rows: int
    intra_period_dates_sample: list[str]
    months_covered: int
    inferred_frequency: str | None
    column_profiles: dict[str, ColumnProfile]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["column_profiles"] = {
            name: asdict(profile) for name, profile in self.column_profiles.items()
        }
        return payload


def discover_raw_files(raw_dir: str | Path) -> list[Path]:
    """All CSV files in ``raw_dir`` (sorted, DVC pointer files excluded)."""
    directory = Path(raw_dir)
    if not directory.exists():
        raise IngestionError(f"raw data directory does not exist: {directory}")
    files = sorted(path for path in directory.glob("*.csv") if path.is_file())
    if not files:
        raise IngestionError(
            f"no CSV files found in {directory}. Run `make data` or "
            f"`python scripts/download_data.py` to fetch the Kaggle dataset."
        )
    return files


def load_raw_frame(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV as strings, preserving the published column names."""
    resolved = Path(path)
    if not resolved.exists():
        raise IngestionError(f"raw dataset not found: {resolved}")
    try:
        frame = pd.read_csv(resolved, dtype=str, keep_default_na=True)
    except (pd.errors.ParserError, OSError) as exc:
        raise IngestionError(f"could not read {resolved}: {exc}") from exc
    if frame.empty:
        raise IngestionError(f"raw dataset is empty: {resolved}")

    # Trim surrounding whitespace without altering the published names.
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    # FRED uses "." for "no observation"; treat it as missing, not as a value.
    frame = frame.replace({".": np.nan})
    logger.info(
        "loaded raw dataset",
        extra={"event": "raw_loaded", "path": str(resolved), "rows": len(frame)},
    )
    return frame


def build_dates(frame: pd.DataFrame, date_columns: list[str]) -> pd.Series:
    """Construct a datetime series from separate year/month/day columns.

    Raises:
        DataValidationError: when a date is invalid (e.g. 2017-02-30) or a date
            component is missing.
    """
    missing = [column for column in date_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"date columns missing from raw dataset: {missing}")

    parts = frame[date_columns].apply(pd.to_numeric, errors="coerce")
    dates = pd.to_datetime(
        {
            "year": parts.iloc[:, 0],
            "month": parts.iloc[:, 1],
            "day": parts.iloc[:, 2],
        },
        errors="coerce",
    )

    invalid = dates.isna()
    if invalid.any():
        samples = frame.loc[invalid, date_columns].head(5).to_dict("records")
        raise DataValidationError(
            f"{int(invalid.sum())} rows have an invalid or missing date "
            f"(first examples: {samples})",
            failures=["invalid_dates"],
        )
    return pd.Series(dates.to_numpy(), index=frame.index, name="date")


def coerce_numeric_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse a published series to ``float``, treating blanks as missing.

    The CSV loader already turns ``""`` into ``NaN``, but frames can also arrive
    from an API call, a test fixture or a FRED download where the missing marker
    is ``"."``. Both are normalised here so the pipeline behaves identically for
    every source.

    Returns:
        ``(parsed, unparseable_mask)`` where the mask flags non-blank values that
        could not be converted to a number.
    """
    if pd.api.types.is_numeric_dtype(series):
        parsed = series.astype(float)
        return parsed, pd.Series(False, index=series.index, dtype=bool)

    cleaned = series.map(lambda value: value.strip() if isinstance(value, str) else value)
    cleaned = cleaned.replace({"": None, ".": None})
    parsed = pd.to_numeric(cleaned, errors="coerce").astype(float)
    unparseable = cleaned.notna() & parsed.isna()
    return parsed, unparseable.astype(bool)


def infer_frequency(dates: pd.Series) -> str | None:
    """Infer the observation cadence from the median spacing of unique dates."""
    unique = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    if len(unique) < 2:
        return None
    median_gap = float(np.median(unique.diff().dropna().dt.total_seconds() / 86_400))
    for threshold, label in FREQUENCY_LABELS:
        if median_gap <= threshold:
            return label
    return "irregular"


def profile_raw_dataset(frame: pd.DataFrame, cfg: DatasetConfig) -> RawDatasetProfile:
    """Profile the raw file without assuming its schema beyond the date columns."""
    dates = build_dates(frame, cfg.date_columns)
    path = str(cfg.raw_path)

    duplicate_dates = sorted(
        pd.Series(dates[dates.duplicated(keep=False)].astype(str).unique()).tolist()
    )
    intra_mask = dates.dt.day != 1
    intra_dates = sorted(dates[intra_mask].dt.strftime("%Y-%m-%d").unique().tolist())

    # Numeric series are profiled by attempting coercion; text stays text.
    profiles: dict[str, ColumnProfile] = {}
    for column in frame.columns:
        if column in cfg.date_columns:
            continue
        raw = frame[column]
        parsed = pd.to_numeric(raw, errors="coerce")
        non_numeric = raw.notna() & parsed.isna()
        non_null = int(parsed.notna().sum())
        kind = "float"
        if non_null == 0:
            kind = "empty"
        elif not non_numeric.any() and bool((parsed.dropna() % 1 == 0).all()):
            kind = "integer"

        first_index = parsed.first_valid_index()
        profiles[column] = ColumnProfile(
            name=column,
            non_null=non_null,
            null_fraction=round(1.0 - non_null / max(len(frame), 1), 4),
            kind=kind,
            minimum=float(parsed.min()) if non_null else None,
            maximum=float(parsed.max()) if non_null else None,
            mean=round(float(parsed.mean()), 4) if non_null else None,
            distinct_values=int(parsed.nunique()) if non_null else 0,
            first_observation=str(dates.loc[first_index].date())
            if first_index is not None
            else None,
            non_numeric_samples=sorted(raw[non_numeric].unique().tolist())[:5],
        )

    return RawDatasetProfile(
        path=path,
        n_rows=int(len(frame)),
        n_columns=int(frame.shape[1]),
        columns=list(frame.columns),
        date_range=((str(dates.min().date()), str(dates.max().date())) if len(dates) else None),
        n_unique_dates=int(dates.nunique()),
        duplicate_dates=duplicate_dates,
        n_intra_period_rows=int(intra_mask.sum()),
        intra_period_dates_sample=intra_dates[:10],
        months_covered=int(dates.dt.to_period("M").nunique()),
        inferred_frequency=infer_frequency(dates),
        column_profiles=profiles,
    )
