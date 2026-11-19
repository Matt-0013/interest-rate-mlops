"""Dataset discovery, retrieval and structural profiling."""

from interest_rate_mlops.ingestion.loader import (
    ColumnProfile,
    RawDatasetProfile,
    build_dates,
    coerce_numeric_series,
    discover_raw_files,
    infer_frequency,
    load_raw_frame,
    profile_raw_dataset,
)
from interest_rate_mlops.ingestion.sources import fetch_fred_series, fetch_kaggle_dataset

__all__ = [
    "ColumnProfile",
    "RawDatasetProfile",
    "build_dates",
    "coerce_numeric_series",
    "discover_raw_files",
    "fetch_fred_series",
    "fetch_kaggle_dataset",
    "infer_frequency",
    "load_raw_frame",
    "profile_raw_dataset",
]
