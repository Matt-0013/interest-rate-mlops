"""Generate the small synthetic fixture used by tests and pipeline CI.

This file is **not** a substitute for the real dataset: it exists so unit tests
and the CI pipeline check can run in seconds without the historical snapshot.

Run: ``python scripts/make_test_fixture.py``
     ``python scripts/make_test_fixture.py --output data/raw/.../index.csv --force``
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "tests" / "fixtures" / "sample_raw.csv"

COLUMNS = [
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


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    year, month = 1958, 1
    step = 0
    while (year, month) <= (1975, 12):
        rate = 4.0 + 0.02 * step + (0.5 if step % 7 == 0 else 0.0)
        unemployment = 5.0 + 0.5 * ((step % 12) / 12)
        inflation = 2.0 + 0.3 * ((step % 6) / 6)
        gdp = "" if step % 3 else f"{3.0 + 0.1 * (step % 5):.1f}"
        target = f"{rate + 0.25:.2f}" if step >= 60 else ""
        upper = "0.25" if step >= 150 else ""
        lower = "0.00" if step >= 150 else ""
        rows.append(
            {
                "Year": str(year),
                "Month": f"{month:02d}",
                "Day": "01",
                "Federal Funds Target Rate": target,
                "Federal Funds Upper Target": upper,
                "Federal Funds Lower Target": lower,
                "Effective Federal Funds Rate": f"{rate:.2f}",
                "Real GDP (Percent Change)": gdp,
                "Unemployment Rate": f"{unemployment:.2f}",
                "Inflation Rate": f"{inflation:.2f}",
            }
        )
        step += 1
        month += 1
        if month == 13:
            year, month = year + 1, 1

    # Intra-month policy-decision rows: only a target rate, never a monthly
    # observation. They must be dropped by the panel builder.
    rows.append(
        {
            "Year": "1970",
            "Month": "06",
            "Day": "17",
            "Federal Funds Target Rate": "9.50",
            "Federal Funds Upper Target": "",
            "Federal Funds Lower Target": "",
            "Effective Federal Funds Rate": "",
            "Real GDP (Percent Change)": "",
            "Unemployment Rate": "",
            "Inflation Rate": "",
        }
    )
    rows.append(
        {
            "Year": "1971",
            "Month": "02",
            "Day": "23",
            "Federal Funds Target Rate": "9.00",
            "Federal Funds Upper Target": "",
            "Federal Funds Lower Target": "",
            "Effective Federal Funds Rate": "",
            "Real GDP (Percent Change)": "",
            "Unemployment Rate": "",
            "Inflation Rate": "",
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_TARGET))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file")
    args = parser.parse_args()

    target = Path(args.output)
    if target.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {target} without --force")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {target}")


if __name__ == "__main__":
    main()
