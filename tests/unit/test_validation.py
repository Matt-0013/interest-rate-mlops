"""Validation behaviour: hard gates fail loudly, soft findings are reported."""

from __future__ import annotations

import pandas as pd
import pytest

from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.preprocessing import build_monthly_panel
from interest_rate_mlops.validation import (
    detect_outliers,
    validate_panel,
    validate_raw_dataset,
)


@pytest.fixture()
def panel(raw_frame, dataset_config):
    """The real panel contract: configured slugs, 216 monthly rows."""
    return build_monthly_panel(raw_frame, dataset_config).panel


def test_valid_raw_dataset_passes(raw_frame, dataset_config):
    report, outliers = validate_raw_dataset(raw_frame, dataset_config)
    assert report.passed, report.failures
    assert not report.failures
    assert any(item.n_outliers == 0 for item in outliers)
    # Informational checks are reported but never block.
    names = {check.name for check in report.checks}
    assert {"required_columns", "valid_dates", "duplicate_records"} <= names
    assert "missingness::Real GDP (Percent Change)" in names


def test_missing_required_column_fails(raw_frame, dataset_config):
    broken = raw_frame.drop(columns=["Inflation Rate"])
    report, _ = validate_raw_dataset(broken, dataset_config)
    assert not report.passed
    assert "required_columns" in report.failures
    with pytest.raises(DataValidationError):
        report.raise_if_failed()


def test_unexpected_column_fails(raw_frame, dataset_config):
    drifted = raw_frame.assign(**{"New Treasury Series": "1.0"})
    report, _ = validate_raw_dataset(drifted, dataset_config)
    assert not report.passed
    assert "unexpected_columns" in report.failures


def test_duplicate_records_fail(raw_frame, dataset_config):
    duplicated = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    report, _ = validate_raw_dataset(duplicated, dataset_config)
    assert "duplicate_records" in report.failures


def test_invalid_dates_fail(raw_frame, dataset_config):
    broken = raw_frame.copy()
    broken.loc[0, "Month"] = "13"
    report, _ = validate_raw_dataset(broken, dataset_config)
    assert "valid_dates" in report.failures


def test_non_numeric_values_fail(raw_frame, dataset_config):
    broken = raw_frame.copy()
    broken.loc[5, "Effective Federal Funds Rate"] = "n/a"
    report, _ = validate_raw_dataset(broken, dataset_config)
    assert not report.passed
    assert "numeric_values" in report.failures


def test_target_out_of_range_fails(raw_frame, dataset_config):
    broken = raw_frame.copy()
    broken.loc[3, "Effective Federal Funds Rate"] = "99.0"
    report, _ = validate_raw_dataset(broken, dataset_config)
    assert "target_range" in report.failures


def test_excessive_target_missingness_fails(dataset_config):
    frame = pd.DataFrame(
        {
            "Year": [str(2000)] * 10,
            "Month": [f"{month:02d}" for month in range(1, 11)],
            "Day": ["01"] * 10,
            "Federal Funds Target Rate": [""] * 10,
            "Federal Funds Upper Target": [""] * 10,
            "Federal Funds Lower Target": [""] * 10,
            "Effective Federal Funds Rate": [""] * 9 + ["5.0"],
            "Real GDP (Percent Change)": [""] * 10,
            "Unemployment Rate": ["4.0"] * 10,
            "Inflation Rate": ["2.0"] * 10,
        }
    )
    report, _ = validate_raw_dataset(frame, dataset_config)
    assert "target_missingness" in report.failures


def test_detect_outliers_flags_extreme_value():
    series = pd.Series([1.0] * 20 + [99.0])
    result = detect_outliers(series, "rate", iqr_factor=3.0)
    assert result.n_outliers == 1
    assert "99" in result.samples[0]


def test_detect_outliers_handles_empty_series():
    result = detect_outliers(pd.Series([None, None], dtype="float64"), "rate")
    assert result.n_outliers == 0


def test_validate_panel_accepts_monthly_panel(panel, dataset_config):
    report = validate_panel(panel, dataset_config)
    assert report.passed, report.failures


def test_validate_panel_rejects_missing_target(panel, dataset_config):
    panel = panel.copy()
    panel.iloc[5, 0] = float("nan")
    report = validate_panel(panel, dataset_config)
    assert not report.passed
    assert "panel_schema" in report.failures


def test_validate_panel_rejects_daily_index(panel, dataset_config):
    index = pd.date_range("2020-01-01", periods=30, freq="D", name="date")
    daily = pd.DataFrame({column: 1.0 for column in panel.columns}, index=index)
    report = validate_panel(daily, dataset_config)
    assert not report.passed


def test_validate_panel_rejects_extra_columns(panel, dataset_config):
    drifted = panel.copy()
    drifted["sneaky_feature"] = 1.0
    report = validate_panel(drifted, dataset_config)
    assert not report.passed
