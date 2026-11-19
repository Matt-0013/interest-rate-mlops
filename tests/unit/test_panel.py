"""Panel construction: observation calendar, intra-period rows, gaps, as-of fills."""

from __future__ import annotations

import pandas as pd
import pytest

from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.ingestion import load_raw_frame
from interest_rate_mlops.preprocessing import build_monthly_panel, load_panel, save_panel


def _panel(raw_frame, dataset_config):
    return build_monthly_panel(raw_frame, dataset_config)


def test_panel_drops_intra_period_rows(raw_frame, dataset_config):
    result = _panel(raw_frame, dataset_config)
    panel, report = result.panel, result.report

    assert report.rows_in == 218
    assert report.intra_period_rows_dropped == 2
    assert report.rows_out == len(panel) == 216
    assert (panel.index.day == 1).all()
    assert panel.index.is_unique
    assert report.start == "1958-01-01"
    assert report.end == "1975-12-01"


def test_panel_uses_configured_slugs(raw_frame, dataset_config):
    panel = _panel(raw_frame, dataset_config).panel
    assert list(panel.columns) == [
        "effective_federal_funds_rate",
        "unemployment_rate",
        "inflation_rate",
        "real_gdp_pct_change",
    ]
    assert panel["effective_federal_funds_rate"].notna().all()


def test_panel_does_not_forward_fill_target(raw_frame, dataset_config):
    panel = _panel(raw_frame, dataset_config).panel
    report = _panel(raw_frame, dataset_config).report
    assert report.asof_fills.get("effective_federal_funds_rate") is None
    assert panel["effective_federal_funds_rate"].isna().sum() == 0


def test_panel_quarterly_series_is_carried_forward(raw_frame, dataset_config):
    panel = _panel(raw_frame, dataset_config).panel
    report = _panel(raw_frame, dataset_config).report
    # GDP is quarterly: 3 of every 4 months are filled as-of.
    assert report.asof_fills["real_gdp_pct_change"] > 0
    assert panel["real_gdp_pct_change"].notna().sum() == len(panel)


def test_panel_records_interior_target_gap_as_error(dataset_config):
    raw = pd.DataFrame(
        {
            "Year": ["2020", "2020", "2020"],
            "Month": ["01", "02", "03"],
            "Day": ["01", "01", "01"],
            "Federal Funds Target Rate": ["", "", ""],
            "Federal Funds Upper Target": ["", "", ""],
            "Federal Funds Lower Target": ["", "", ""],
            "Effective Federal Funds Rate": ["1.5", "", "1.6"],
            "Real GDP (Percent Change)": ["", "", ""],
            "Unemployment Rate": ["4.0", "4.1", "4.2"],
            "Inflation Rate": ["2.0", "2.0", "2.1"],
        }
    )
    with pytest.raises(DataValidationError, match="interior missing months"):
        build_monthly_panel(raw, dataset_config)


def test_panel_allows_gap_when_contiguity_not_required(dataset_config):
    relaxed = dataset_config.model_copy(
        update={
            "panel": dataset_config.panel.model_copy(update={"require_contiguous_target": False})
        }
    )
    raw = pd.DataFrame(
        {
            "Year": ["2020", "2020", "2020"],
            "Month": ["01", "02", "03"],
            "Day": ["01", "01", "01"],
            "Federal Funds Target Rate": ["", "", ""],
            "Federal Funds Upper Target": ["", "", ""],
            "Federal Funds Lower Target": ["", "", ""],
            "Effective Federal Funds Rate": ["1.5", "", "1.6"],
            "Real GDP (Percent Change)": ["", "", ""],
            "Unemployment Rate": ["4.0", "4.1", "4.2"],
            "Inflation Rate": ["2.0", "2.0", "2.1"],
        }
    )
    result = build_monthly_panel(raw, relaxed)
    assert result.report.target_gaps == ["2020-02-01"]
    assert result.report.has_target_gaps


def test_panel_rejects_multiple_observations_per_month(dataset_config):
    raw = pd.DataFrame(
        {
            "Year": ["2020", "2020"],
            "Month": ["01", "01"],
            "Day": ["01", "01"],
            "Federal Funds Target Rate": ["", ""],
            "Federal Funds Upper Target": ["", ""],
            "Federal Funds Lower Target": ["", ""],
            "Effective Federal Funds Rate": ["1.5", "1.6"],
            "Real GDP (Percent Change)": ["", ""],
            "Unemployment Rate": ["4.0", "4.0"],
            "Inflation Rate": ["2.0", "2.0"],
        }
    )
    with pytest.raises(DataValidationError, match="multiple observations"):
        build_monthly_panel(raw, dataset_config)


def test_panel_applies_start_window(raw_frame, dataset_config):
    clipped = dataset_config.model_copy(
        update={"panel": dataset_config.panel.model_copy(update={"start": "1960-01-01"})}
    )
    panel = _panel(raw_frame, clipped).panel
    assert panel.index.min() == pd.Timestamp("1960-01-01")
    assert panel.index.max() == pd.Timestamp("1975-12-01")


def test_panel_rejects_unparseable_series_value(dataset_config):
    raw = load_raw_frame(dataset_config.raw_path)
    raw.loc[2, "Effective Federal Funds Rate"] = "not-a-rate"
    with pytest.raises(DataValidationError, match="non-numeric"):
        build_monthly_panel(raw, dataset_config)


def test_panel_round_trips_through_parquet(raw_frame, dataset_config, tmp_path):
    result = _panel(raw_frame, dataset_config)
    path = save_panel(result.panel, tmp_path / "panel.parquet", result.report.as_dict())
    loaded, metadata = load_panel(path)

    pd.testing.assert_frame_equal(loaded, result.panel, check_freq=False)
    assert metadata["rows_out"] == 216
    assert metadata["target"] == "effective_federal_funds_rate"


def test_panel_rejects_empty_after_window(raw_frame, dataset_config):
    impossible = dataset_config.model_copy(
        update={"panel": dataset_config.panel.model_copy(update={"start": "2050-01-01"})}
    )
    with pytest.raises(DataValidationError, match="empty"):
        build_monthly_panel(raw_frame, impossible)


def test_panel_rejects_missing_configured_series(dataset_config):
    raw = load_raw_frame(dataset_config.raw_path).drop(columns=["Inflation Rate"])
    with pytest.raises(DataValidationError, match="not present in raw data"):
        build_monthly_panel(raw, dataset_config)
