"""Exploratory data analysis.

Produces the analysis that motivated the modelling decisions: persistence of the
rate series, regime shifts, missingness of the sparse macro series, and the
autocorrelation structure that justifies lag features. All numbers in the report
are computed from the data at run time - nothing is hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from interest_rate_mlops.logging_utils import get_logger  # noqa: E402
from interest_rate_mlops.storage import write_json  # noqa: E402

logger = get_logger(__name__)


def autocorrelation(series: pd.Series, lag: int) -> float:
    """Pearson autocorrelation at ``lag`` (sample, pairwise-complete)."""
    if lag <= 0 or len(series) <= lag:
        return float("nan")
    left = series.iloc[:-lag].to_numpy(dtype=float)
    right = series.iloc[lag:].to_numpy(dtype=float)
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def summarise_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-series descriptive statistics including persistence measures."""
    rows = []
    for column in panel.columns:
        series = panel[column].astype(float)
        observed = series.dropna()
        rows.append(
            {
                "series": column,
                "observations": int(observed.size),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "mean": round(float(observed.mean()), 4) if observed.size else None,
                "std": round(float(observed.std(ddof=1)), 4) if observed.size > 1 else None,
                "min": round(float(observed.min()), 4) if observed.size else None,
                "max": round(float(observed.max()), 4) if observed.size else None,
                "first": str(observed.index.min().date()) if observed.size else None,
                "last": str(observed.index.max().date()) if observed.size else None,
                "acf_1": round(autocorrelation(observed, 1), 4) if observed.size > 2 else None,
                "acf_12": round(autocorrelation(observed, 12), 4) if observed.size > 13 else None,
            }
        )
    return pd.DataFrame(rows)


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_history(panel: pd.DataFrame, target: str, path: Path, unit: str) -> None:
    figure, axes = plt.subplots(figsize=(11, 4.5))
    for column in panel.columns:
        style = {"linewidth": 1.8, "color": "#1f3b73"} if column == target else {"alpha": 0.6}
        axes.plot(panel.index, panel[column], label=column, **style)
    axes.set_title("Monthly interest-rate and macro series")
    axes.set_ylabel(unit)
    axes.set_xlabel("observation month")
    axes.grid(alpha=0.25)
    axes.legend(frameon=False, ncol=2, fontsize=8)
    _save(figure, path)


def _plot_distribution(panel: pd.DataFrame, target: str, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.hist(panel[target].dropna(), bins=40, color="#1f3b73", edgecolor="white")
    axes.set_title(f"Distribution of {target}")
    axes.set_xlabel("percent")
    axes.set_ylabel("months")
    axes.grid(alpha=0.2, axis="y")
    _save(figure, path)


def _plot_rolling(panel: pd.DataFrame, target: str, path: Path) -> None:
    series = panel[target]
    figure, axes = plt.subplots(figsize=(11, 4.5))
    axes.plot(series.index, series, label="monthly value", color="#1f3b73", linewidth=1.2)
    axes.plot(series.index, series.rolling(12).mean(), label="12-month mean", color="#c8102e")
    axes.fill_between(
        series.index,
        series.rolling(12).mean() - series.rolling(12).std(),
        series.rolling(12).mean() + series.rolling(12).std(),
        color="#c8102e",
        alpha=0.12,
        label="+/- 1 rolling std",
    )
    axes.set_title("Level and 12-month rolling statistics")
    axes.set_ylabel("percent")
    axes.grid(alpha=0.2)
    axes.legend(frameon=False)
    _save(figure, path)


def _plot_acf(panel: pd.DataFrame, target: str, path: Path, max_lag: int = 36) -> None:
    series = panel[target].dropna()
    lags = list(range(1, max_lag + 1))
    values = [autocorrelation(series, lag) for lag in lags]
    figure, axes = plt.subplots(figsize=(9, 3.8))
    axes.bar(lags, values, color="#1f3b73")
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_title(f"Autocorrelation of {target}")
    axes.set_xlabel("lag (months)")
    axes.set_ylabel("correlation")
    axes.grid(alpha=0.2, axis="y")
    _save(figure, path)


def _plot_by_month(panel: pd.DataFrame, target: str, path: Path) -> None:
    frame = panel[[target]].copy()
    frame["month"] = frame.index.month
    grouped = frame.groupby("month")[target].agg(["mean", "std"])
    figure, axes = plt.subplots(figsize=(8, 3.8))
    axes.bar(grouped.index, grouped["mean"], yerr=grouped["std"], color="#1f3b73", alpha=0.85)
    axes.set_title(f"Average {target} by calendar month (seasonality check)")
    axes.set_xlabel("month")
    axes.set_ylabel("mean percent")
    axes.set_xticks(range(1, 13))
    axes.grid(alpha=0.2, axis="y")
    _save(figure, path)


def _markdown(
    summary: pd.DataFrame,
    target: str,
    panel: pd.DataFrame,
    unit: str,
    extra: dict[str, Any],
) -> str:
    table = [
        "| series | obs | missing | mean | std | min | max | acf(1) | acf(12) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.to_dict("records"):
        table.append(
            f"| {row['series']} | {row['observations']} | {row['missing_pct']}% | "
            f"{row['mean']} | {row['std']} | {row['min']} | {row['max']} | "
            f"{row['acf_1']} | {row['acf_12']} |"
        )

    series = panel[target]
    peak_date = series.idxmax()
    trough_date = series[series > 0].idxmin()
    changes = series.diff().abs()

    return "\n".join(
        [
            "# Exploratory data analysis",
            "",
            f"- Panel window: **{panel.index.min().date()} to {panel.index.max().date()}**",
            f"- Monthly observations: **{len(panel)}**",
            f"- Target series: **{target}** ({unit})",
            f"- Peak: **{series.max():.2f}%** in {peak_date.date()}",
            f"- Trough (non-zero): **{series[series > 0].min():.2f}%** in {trough_date.date()}",
            f"- Largest single-month move: **{changes.max():.2f} pp** "
            f"in {changes.idxmax().date()}",
            f"- Autocorrelation at lag 1: **{autocorrelation(series, 1):.3f}**, "
            f"at lag 12: **{autocorrelation(series, 12):.3f}**",
            "",
            "## Series summary",
            "",
            *table,
            "",
            "## What this implies for modelling",
            "",
            "- The target is highly persistent (lag-1 autocorrelation close to 1), so lag and "
            "rolling-window features carry most of the signal; a naive 'carry forward' "
            "baseline is strong and must be beaten honestly.",
            "- The series spans very different policy regimes (double-digit rates in the "
            "late 1970s/early 1980s, a zero lower bound after 2008). Fold-to-fold error "
            "variation is therefore expected and is reported rather than hidden.",
            "- The target is observed monthly, not daily. Lags are expressed in "
            "observations (months), never in calendar days.",
            "- `Real GDP (Percent Change)` is quarterly and is carried forward as the last "
            "published value; missing months are periods with no scheduled observation, "
            "not data errors.",
            "- Rates near zero make MAPE unstable, which is why MAE/RMSE are the primary "
            "metrics and MAPE is reported with an explicit coverage figure.",
            "",
            "## Figures",
            "",
            "![history](target_history.png)",
            "",
            "![distribution](target_distribution.png)",
            "",
            "![rolling](rolling_statistics.png)",
            "",
            "![acf](autocorrelation.png)",
            "",
            "![seasonality](seasonality_by_month.png)",
            "",
            f"Data-quality context: {extra.get('raw_rows', 'n/a')} raw rows reduced to "
            f"{len(panel)} monthly observations after dropping "
            f"{extra.get('intra_period_rows', 'n/a')} intra-month policy-decision rows.",
            "",
        ]
    )


def build_eda_report(
    panel: pd.DataFrame,
    output_dir: str | Path,
    *,
    target: str,
    unit: str = "percent",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write figures, a summary JSON and a markdown EDA report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    extra = extra or {}

    summary = summarise_panel(panel)
    figures = {
        "target_history": str(_plot_history(panel, target, directory / "target_history.png", unit)),
        "target_distribution": str(
            _plot_distribution(panel, target, directory / "target_distribution.png")
        ),
        "rolling_statistics": str(
            _plot_rolling(panel, target, directory / "rolling_statistics.png")
        ),
        "autocorrelation": str(_plot_acf(panel, target, directory / "autocorrelation.png")),
        "seasonality_by_month": str(
            _plot_by_month(panel, target, directory / "seasonality_by_month.png")
        ),
    }

    payload = {
        "window": {
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
            "months": int(len(panel)),
        },
        "target": target,
        "unit": unit,
        "summary": summary.to_dict("records"),
        "figures": figures,
        **extra,
    }
    write_json(directory / "eda_summary.json", payload)
    (directory / "eda_report.md").write_text(
        _markdown(summary, target, panel, unit, extra), encoding="utf-8"
    )
    logger.info("EDA report written", extra={"event": "eda_complete", "path": str(directory)})
    return payload
