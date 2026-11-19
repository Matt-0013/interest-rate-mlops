"""Raw dataset loading and profiling."""

from __future__ import annotations

import pandas as pd
import pytest

from interest_rate_mlops.exceptions import DataValidationError, IngestionError
from interest_rate_mlops.ingestion import (
    build_dates,
    discover_raw_files,
    infer_frequency,
    profile_raw_dataset,
)


def test_discover_raw_files_returns_csv(project_root):
    files = discover_raw_files(project_root / "tests" / "fixtures")
    assert [path.name for path in files] == ["sample_raw.csv"]


def test_discover_raw_files_empty_dir(tmp_path):
    with pytest.raises(IngestionError, match="no CSV files"):
        discover_raw_files(tmp_path)


def test_load_raw_frame_preserves_published_columns(raw_frame):
    expected = [
        "Year",
        "Month",
        "Day",
        "Federal Funds Target Rate",
        "Federal Funds Upper Target",
        "Federal Funds Lower Target",
        "Effective Federal Funds Rate",
        "Real GDP (Percent Change)",
        "Unemployment Rate",
        "Inflation Rate",
    ]
    assert list(raw_frame.columns) == expected
    assert len(raw_frame) == 218  # 216 monthly rows + 2 intra-month rows


def test_build_dates_parses_components(raw_frame):
    dates = build_dates(raw_frame, ["Year", "Month", "Day"])
    assert dates.dtype.kind == "M"
    assert dates.min() == pd.Timestamp("1958-01-01")
    assert dates.max() == pd.Timestamp("1975-12-01")


def test_build_dates_rejects_invalid_dates():
    frame = pd.DataFrame({"Year": ["2019"], "Month": ["02"], "Day": ["30"]})
    with pytest.raises(DataValidationError, match="invalid or missing date"):
        build_dates(frame, ["Year", "Month", "Day"])


def test_build_dates_rejects_missing_columns(raw_frame):
    with pytest.raises(DataValidationError, match="date columns missing"):
        build_dates(raw_frame.drop(columns=["Day"]), ["Year", "Month", "Day"])


def test_infer_frequency_monthly(raw_frame):
    dates = build_dates(raw_frame, ["Year", "Month", "Day"])
    assert infer_frequency(dates) == "monthly"


def test_infer_frequency_handles_single_observation():
    assert infer_frequency(pd.Series([pd.Timestamp("2020-01-01")])) is None


def test_profile_detects_intra_period_rows_and_frequency(raw_frame, dataset_config):
    profile = profile_raw_dataset(raw_frame, dataset_config)

    assert profile.n_rows == 218
    assert profile.n_columns == 10
    assert profile.n_intra_period_rows == 2
    assert "1970-06-17" in profile.intra_period_dates_sample
    assert profile.months_covered == 216
    assert profile.inferred_frequency == "monthly"
    assert profile.duplicate_dates == []
    assert profile.date_range == ("1958-01-01", "1975-12-01")

    effective = profile.column_profiles["Effective Federal Funds Rate"]
    assert effective.kind == "float"
    assert effective.non_null == 216
    assert effective.first_observation == "1958-01-01"

    gdp = profile.column_profiles["Real GDP (Percent Change)"]
    assert gdp.non_null == 72  # quarterly


def test_profile_reports_first_observation_of_sparse_series(raw_frame, dataset_config):
    profile = profile_raw_dataset(raw_frame, dataset_config)
    target_rate = profile.column_profiles["Federal Funds Target Rate"]
    assert target_rate.first_observation == "1963-01-01"
    assert target_rate.non_null > 0
