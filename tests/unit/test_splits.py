"""Time-series split invariants: no shuffling, no overlap, future-only holdout."""

from __future__ import annotations

import pandas as pd
import pytest

from interest_rate_mlops.config import CVConfig, SplitConfig
from interest_rate_mlops.exceptions import DataValidationError
from interest_rate_mlops.training.splits import chronological_split, walk_forward_splits


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-01", periods=n, freq="MS")


def test_chronological_split_is_future_holdout():
    split = chronological_split(
        _index(100), SplitConfig(test_fraction=0.2, min_test_observations=5)
    )
    assert split.n_train == 80
    assert split.n_test == 20
    assert split.train_idx.max() < split.test_idx.min()
    assert split.train_end == "2006-08-01"
    assert split.test_start == "2006-09-01"


def test_chronological_split_respects_min_test_size():
    split = chronological_split(_index(20), SplitConfig(test_fraction=0.1, min_test_observations=5))
    assert split.n_test == 5


def test_chronological_split_rejects_tiny_datasets():
    with pytest.raises(DataValidationError):
        chronological_split(_index(3), SplitConfig(test_fraction=0.2, min_test_observations=2))


def test_split_description_is_serialisable():
    split = chronological_split(_index(50), SplitConfig(test_fraction=0.2, min_test_observations=5))
    payload = split.as_dict()
    assert payload["n_train"] + payload["n_test"] == 50
    assert "train_idx" not in payload


def test_walk_forward_folds_expand_and_do_not_overlap():
    folds = walk_forward_splits(100, CVConfig(n_splits=4, gap=0))
    assert len(folds) == 4
    previous_end = 0
    for train_index, test_index in folds:
        assert train_index.max() < test_index.min()
        assert train_index.min() == 0  # expanding window
        assert train_index.max() >= previous_end
        previous_end = test_index.max()


def test_walk_forward_gap_creates_separation():
    folds = walk_forward_splits(100, CVConfig(n_splits=3, gap=2))
    for train_index, test_index in folds:
        assert test_index.min() - train_index.max() > 2


def test_walk_forward_rejects_short_training_blocks():
    with pytest.raises(DataValidationError):
        walk_forward_splits(4, CVConfig(n_splits=5, gap=0))
