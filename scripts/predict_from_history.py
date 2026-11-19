"""Call the deployed API with a raw monthly history.

The API rebuilds the model's features from the history with the same code used in
training, so no feature engineering happens in this client.

Usage:
    python scripts/predict_from_history.py --csv history.csv --api-url http://localhost:8000

The CSV needs ``observation_date,value`` columns with month-start dates.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV with observation_date,value")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if not {"observation_date", "value"}.issubset(frame.columns):
        print("error: CSV must contain observation_date and value columns", file=sys.stderr)
        return 2

    history = [
        {"date": pd.Timestamp(row.observation_date).date().isoformat(), "value": float(row.value)}
        for row in frame.itertuples()
    ]
    base = args.api_url.rstrip("/")

    response = requests.post(f"{base}/predict", json={"history": history}, timeout=args.timeout)
    if response.status_code != 200:
        print(f"error {response.status_code}: {response.text}", file=sys.stderr)
        return 1

    body = response.json()
    print(
        json.dumps(
            {
                "prediction": body["prediction"],
                "unit": body["unit"],
                "model": f"{body['model_name']} v{body['model_version']}",
                "origin_date": body["origin_date"],
                "request_id": body["request_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
