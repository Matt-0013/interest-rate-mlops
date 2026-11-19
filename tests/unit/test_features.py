"""Feature engineering correctness and leakage prevention.

The numeric expectations below are the contract: ``lag_1`` is the freshest known
observation and every rolling statistic ends at the origin month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_panel

from interest_rate_mlops.config import FeatureConfig
from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.features import (
    build_feature_row,
    build_features_frame,
    build_supervised_features,
    history_to_panel,
    required_history_length,
    validate_feature_payload,
)


def _linear_panel(n: int = 20) -> pd.DataFrame:
    index = pd.date_range("2010-01-01", periods=n, freq="MS", name="date")
    return pd.DataFrame({"target": np.arange(1.0, n + 1.0)}, index=index)


def test_lag_semantics_are_k_steps_before_target():
    panel = _linear_panel()
    features = build_features_frame(
        panel,
        FeatureConfig(
            lags=[1, 2, 3],
            rolling_windows=[3],
            rolling_stats=["mean"],
            diffs=[1],
            min_observations=4,
        ),
    )
    origin = panel.index[15]  # value 16
    assert features.loc[origin, "lag_1"] == 16.0
    assert features.loc[origin, "lag_2"] == 15.0
    assert features.loc[origin, "lag_3"] == 14.0


def test_rolling_statistics_are_backward_looking():
    panel = _linear_panel()
    features = build_features_frame(
        panel,
        FeatureConfig(
            lags=[1],
            rolling_windows=[3],
            rolling_stats=["mean", "std"],
            diffs=[1],
            min_observations=3,
        ),
    )
    origin = panel.index[15]
    window = [14.0, 15.0, 16.0]
    assert features.loc[origin, "roll_mean_3"] == pytest.approx(np.mean(window))
    assert features.loc[origin, "roll_std_3"] == pytest.approx(np.std(window, ddof=1))


def test_diff_is_difference_from_origin():
    panel = _linear_panel()
    features = build_features_frame(
        panel,
        FeatureConfig(
            lags=[1], rolling_windows=[3], rolling_stats=["mean"], diffs=[1, 6], min_observations=7
        ),
    )
    origin = panel.index[15]
    assert features.loc[origin, "diff_1"] == pytest.approx(1.0)
    assert features.loc[origin, "diff_6"] == pytest.approx(6.0)


def test_burn_in_rows_are_nan_not_partial_windows():
    panel = _linear_panel()
    features = build_features_frame(
        panel,
        FeatureConfig(
            lags=[12], rolling_windows=[3], rolling_stats=["mean"], diffs=[1], min_observations=13
        ),
    )
    assert features["lag_12"].iloc[:11].isna().all()
    assert features["lag_12"].iloc[11:].notna().all()
    assert features["roll_mean_3"].iloc[:2].isna().all()


def test_target_is_next_observation_and_aligned():
    panel = _linear_panel()
    matrix = build_supervised_features(
        panel,
        FeatureConfig(
            lags=[1, 2], rolling_windows=[3], rolling_stats=["mean"], diffs=[1], min_observations=3
        ),
    )
    assert matrix.X.index[0] == panel.index[2]
    assert matrix.y.iloc[0] == 4.0
    assert matrix.target_dates.iloc[0] == panel.index[3]
    assert (pd.DatetimeIndex(matrix.target_dates) > matrix.origins).all()
    assert list(matrix.feature_names) == ["lag_1", "lag_2", "roll_mean_3", "diff_1"]


def test_no_future_leakage_when_future_values_change():
    panel = _linear_panel(30)
    cfg = FeatureConfig(
        lags=[1, 3],
        rolling_windows=[3],
        rolling_stats=["mean", "std"],
        diffs=[1],
        min_observations=4,
    )
    baseline = build_features_frame(panel, cfg)

    mutated = panel.copy()
    cutoff = panel.index[20]
    mutated.loc[mutated.index > cutoff, "target"] = 999.0
    after = build_features_frame(mutated, cfg)

    pd.testing.assert_frame_equal(baseline.loc[:cutoff], after.loc[:cutoff])


def test_missing_observation_drops_rows_instead_of_imputing():
    panel = _linear_panel(20)
    panel.iloc[10, 0] = np.nan
    with pytest.raises(DataValidationError, match="missing observations"):
        build_supervised_features(
            panel,
            FeatureConfig(
                lags=[1], rolling_windows=[3], rolling_stats=["mean"], diffs=[1], min_observations=3
            ),
        )


def test_preprocessing_rejects_gaps_before_features(dataset_config, raw_frame):
    # The panel builder is the stage that refuses interior target gaps.
    from interest_rate_mlops.preprocessing import build_monthly_panel

    raw = raw_frame.copy()
    mask = (raw["Year"] == "1960") & (raw["Month"] == "05")
    raw.loc[mask, "Effective Federal Funds Rate"] = ""
    with pytest.raises(DataValidationError):
        build_monthly_panel(raw, dataset_config)


def test_insufficient_history_raises():
    cfg = FeatureConfig(
        lags=[12], rolling_windows=[12], rolling_stats=["mean"], diffs=[1], min_observations=13
    )
    assert required_history_length(cfg) == 12
    with pytest.raises(DataValidationError, match="short|history"):
        build_supervised_features(make_panel(6), cfg)


def test_feature_row_matches_last_supervised_row():
    panel = _linear_panel(20)
    cfg = FeatureConfig(
        lags=[1, 3], rolling_windows=[3], rolling_stats=["mean"], diffs=[1], min_observations=4
    )
    origin, row = build_feature_row(panel, cfg)

    # The supervised matrix stops one month earlier: the freshest origin has no
    # future observation to learn from, but it is exactly what serving uses.
    features = build_features_frame(panel, cfg).dropna()
    assert origin == features.index[-1] == panel.index[-1]
    for name, value in row.items():
        assert value == pytest.approx(features.loc[origin, name])


# ---------------------------------------------------------------------------
# Serving-time validation
# ---------------------------------------------------------------------------
def test_validate_feature_payload_orders_columns():
    frame = validate_feature_payload(
        {"b": 2.0, "a": 1.0, "c": 3.0}, ["a", "b", "c"], allowed_range=(0.0, 25.0)
    )
    assert list(frame.columns) == ["a", "b", "c"]
    assert frame.iloc[0].tolist() == [1.0, 2.0, 3.0]


def test_validate_feature_payload_rejects_missing_and_extra():
    with pytest.raises(DataValidationError, match="missing required features"):
        validate_feature_payload({"a": 1.0}, ["a", "b"])
    with pytest.raises(DataValidationError, match="unexpected features"):
        validate_feature_payload({"a": 1.0, "b": 2.0, "z": 3.0}, ["a", "b"])


def test_validate_feature_payload_rejects_bad_values():
    with pytest.raises(DataValidationError, match="must be a number"):
        validate_feature_payload({"a": "1.0"}, ["a"])
    with pytest.raises(DataValidationError, match="finite"):
        validate_feature_payload({"a": float("nan")}, ["a"])
    with pytest.raises(DataValidationError, match="plausible band"):
        validate_feature_payload({"lag_1": 99.0}, ["lag_1"], allowed_range=(0.0, 25.0))


def test_history_to_panel_requires_month_starts_and_contiguity():
    with pytest.raises(DataValidationError, match="month-start"):
        history_to_panel(
            [{"date": "2020-01-15", "value": 1.0}, {"date": "2020-02-01", "value": 1.1}]
        )

    with pytest.raises(DataValidationError, match="contiguous"):
        history_to_panel(
            [
                {"date": "2020-01-01", "value": 1.0},
                {"date": "2020-03-01", "value": 1.2},
            ]
        )


def test_history_to_panel_rejects_duplicates():
    with pytest.raises(DataValidationError, match="duplicate"):
        history_to_panel(
            [
                {"date": "2020-01-01", "value": 1.0},
                {"date": "2020-01-01", "value": 1.1},
            ]
        )


def test_features_from_history_matches_direct_computation():
    from interest_rate_mlops.features.serving import features_from_history

    panel = _linear_panel(20)
    cfg = FeatureConfig(
        lags=[1, 2], rolling_windows=[3], rolling_stats=["mean"], diffs=[1], min_observations=3
    )
    observations = [
        {"date": index.strftime("%Y-%m-%d"), "value": float(value)}
        for index, value in panel["target"].items()
    ]
    result = features_from_history(observations, cfg)
    direct = build_features_frame(panel, cfg).dropna().iloc[-1]

    assert result["observations_used"] == 20
    for name in cfg.names():
        assert result["features"][name] == pytest.approx(float(direct[name]))
