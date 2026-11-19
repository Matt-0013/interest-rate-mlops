"""Time-series splitting.

Random splits are invalid here: a shuffled split lets the model interpolate
between neighbouring months and reports accuracy that does not exist at
forecast time. Both splitters below preserve time order and never let a test
month appear in its own training window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from interest_rate_mlops.config import CVConfig, SplitConfig
from interest_rate_mlops.exceptions import DataValidationError


@dataclass(frozen=True)
class Split:
    """Positional train/test indices plus a human-readable description."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_test(self) -> int:
        return int(len(self.test_idx))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({"n_train": self.n_train, "n_test": self.n_test})
        payload.pop("train_idx")
        payload.pop("test_idx")
        return payload


def chronological_split(
    index: pd.DatetimeIndex,
    cfg: SplitConfig,
) -> Split:
    """Hold out the most recent ``test_fraction`` of the sample.

    The test block is the *future* relative to training, which is the only
    honest way to estimate next-month forecast error.
    """
    n_obs = len(index)
    if n_obs < 4:
        raise DataValidationError(f"need at least 4 samples to split, got {n_obs}")

    n_test = max(int(round(n_obs * cfg.test_fraction)), cfg.min_test_observations)
    n_test = min(n_test, n_obs - 2)
    n_train = n_obs - n_test
    if n_train < 2:
        raise DataValidationError(
            f"chronological split leaves {n_train} training rows "
            f"(n_obs={n_obs}, test={n_test}); reduce split.test_fraction"
        )

    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, n_obs)
    return Split(
        train_idx=train_idx,
        test_idx=test_idx,
        train_start=str(index[train_idx[0]].date()),
        train_end=str(index[train_idx[-1]].date()),
        test_start=str(index[test_idx[0]].date()),
        test_end=str(index[test_idx[-1]].date()),
    )


def walk_forward_splits(n_train: int, cfg: CVConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over the training block only.

    Fold ``k`` trains on all months before the fold's test window and scores the
    immediately following months, which mirrors repeated next-month forecasting.

    Args:
        n_train: number of training rows (the holdout is never used here).
        cfg: ``cv.n_splits`` and optional ``gap``.
    """
    if n_train < cfg.n_splits + 2:
        raise DataValidationError(
            f"walk-forward validation needs more than {cfg.n_splits + 1} training rows, "
            f"got {n_train}"
        )

    splitter = TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.gap)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_index, test_index in splitter.split(np.arange(n_train)):
        if len(test_index) == 0:  # pragma: no cover - guarded by n_train check
            continue
        folds.append((train_index, test_index))
    if not folds:  # pragma: no cover - guarded above
        raise DataValidationError("walk-forward validation produced no folds")
    return folds
