"""Evaluation plots.

Every plot is written to disk as an artifact (no interactive backend), so the
same code runs inside Airflow, CI and a container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from interest_rate_mlops.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

PALETTE = {"actual": "#1f3b73", "predicted": "#c8102e", "neutral": "#6b7280"}


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    path: str | Path,
    title: str,
    unit: str = "percent",
) -> Path:
    """Actual vs predicted over the chronological test block."""
    figure, axes = plt.subplots(figsize=(10, 4.5))
    axes.plot(
        y_true.index, y_true.to_numpy(), color=PALETTE["actual"], label="Actual", linewidth=1.6
    )
    axes.plot(
        y_pred.index,
        y_pred.to_numpy(),
        color=PALETTE["predicted"],
        label="Predicted",
        linewidth=1.4,
        linestyle="--",
    )
    axes.set_title(title)
    axes.set_ylabel(unit)
    axes.set_xlabel("origin month")
    axes.grid(alpha=0.25)
    axes.legend(frameon=False)
    return _save(figure, Path(path))


def plot_error_distribution(residuals: pd.Series, path: str | Path, title: str) -> Path:
    """Histogram of residuals (prediction error = actual - predicted)."""
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.hist(residuals.to_numpy(), bins=30, color=PALETTE["neutral"], edgecolor="white")
    axes.axvline(0.0, color=PALETTE["predicted"], linestyle="--", linewidth=1.2)
    axes.set_title(title)
    axes.set_xlabel("residual (percentage points)")
    axes.set_ylabel("count")
    axes.grid(alpha=0.2, axis="y")
    return _save(figure, Path(path))


def plot_residuals(
    predicted: pd.Series, residuals: pd.Series, path: str | Path, title: str
) -> Path:
    """Residuals against predicted value, to reveal bias or heteroscedasticity."""
    figure, axes = plt.subplots(figsize=(6.5, 4.5))
    axes.scatter(
        predicted.to_numpy(), residuals.to_numpy(), s=18, color=PALETTE["neutral"], alpha=0.8
    )
    axes.axhline(0.0, color=PALETTE["predicted"], linestyle="--", linewidth=1.2)
    axes.set_title(title)
    axes.set_xlabel("predicted rate (percent)")
    axes.set_ylabel("residual (percentage points)")
    axes.grid(alpha=0.2)
    return _save(figure, Path(path))


def plot_model_comparison(
    scores: dict[str, dict[str, float | None]],
    path: str | Path,
    metrics: tuple[str, ...] = ("rmse", "mae"),
) -> Path:
    """Grouped bar chart comparing candidates on the held-out test block."""
    names = list(scores)
    positions = np.arange(len(names))
    width = 0.8 / max(len(metrics), 1)

    figure, axes = plt.subplots(figsize=(8, 4.5))
    for offset, metric in enumerate(metrics):
        values = [scores[name].get(metric) or 0.0 for name in names]
        axes.bar(positions + offset * width, values, width=width, label=metric.upper())

    axes.set_xticks(positions + width * (len(metrics) - 1) / 2)
    axes.set_xticklabels(names, rotation=12)
    axes.set_ylabel("error (percentage points, lower is better)")
    axes.set_title("Model comparison on the chronological test block")
    axes.grid(alpha=0.2, axis="y")
    axes.legend(frameon=False)
    return _save(figure, Path(path))


def plot_training_timeline(origins: pd.DatetimeIndex, path: str | Path) -> Path:
    """Show how many supervised samples each origin contributes (sanity check)."""
    figure, axes = plt.subplots(figsize=(9, 3.2))
    axes.plot(origins, np.arange(len(origins)), color=PALETTE["actual"], linewidth=1.4)
    axes.set_title("Origin months covered by the supervised sample")
    axes.set_xlabel("origin month")
    axes.set_ylabel("cumulative samples")
    axes.grid(alpha=0.2)
    return _save(figure, Path(path))


def save_all_diagnostics(
    results: list[Any],
    output_dir: str | Path,
    unit: str = "percent",
) -> dict[str, list[str]]:
    """Write the standard diagnostics for every model result."""
    directory = Path(output_dir)
    written: dict[str, list[str]] = {}
    for result in results:
        written[result.name] = [
            str(
                plot_actual_vs_predicted(
                    result.y_true,
                    result.predictions,
                    directory / f"{result.name}_actual_vs_predicted.png",
                    f"{result.name}: actual vs predicted (test block)",
                    unit=unit,
                )
            ),
            str(
                plot_error_distribution(
                    result.residuals,
                    directory / f"{result.name}_error_distribution.png",
                    f"{result.name}: prediction error distribution",
                )
            ),
            str(
                plot_residuals(
                    result.predictions,
                    result.residuals,
                    directory / f"{result.name}_residuals.png",
                    f"{result.name}: residuals vs predicted",
                )
            ),
        ]
    return written
