"""End-to-end API smoke test.

Checks health, model info, feature retrieval, inference, persistence, monitoring
status, Prometheus metrics and the GraphQL surface. Exits non-zero if any check
fails, so it can gate a deployment.

Usage:
    python scripts/verify_api.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def _check(results: list[tuple[str, bool, str]], name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    base = args.api_url.rstrip("/")
    results: list[tuple[str, bool, str]] = []

    def get(path: str) -> Any:
        return requests.get(f"{base}{path}", timeout=args.timeout)

    try:
        health = get("/health")
        _check(
            results,
            "GET /health",
            health.status_code == 200,
            f"{health.status_code} {health.json().get('status')}",
        )

        model_info = get("/model/info")
        _check(
            results,
            "GET /model/info",
            model_info.status_code == 200,
            f"model v{model_info.json().get('version')}"
            if model_info.status_code == 200
            else str(model_info.status_code),
        )

        features_response = get("/model/features")
        features_ok = features_response.status_code == 200
        _check(
            results,
            "GET /model/features",
            features_ok,
            "" if features_ok else str(features_response.status_code),
        )

        if model_info.status_code == 200 and features_ok:
            payload = {
                "features": features_response.json()["features"],
            }
            predictions_before = get("/predictions?limit=1").json().get("total", 0)

            prediction = requests.post(f"{base}/predict", json=payload, timeout=args.timeout)
            prediction_ok = prediction.status_code == 200
            _check(
                results,
                "POST /predict",
                prediction_ok,
                f"{prediction.json().get('prediction')} {prediction.json().get('unit', '')}"
                if prediction_ok
                else prediction.text[:200],
            )

            predictions_after = get("/predictions?limit=1").json()
            _check(
                results,
                "GET /predictions (persistence)",
                predictions_after.get("total", 0) == predictions_before + 1,
                f"total={predictions_after.get('total')}",
            )

        drift = get("/drift/status")
        _check(
            results,
            "GET /drift/status",
            drift.status_code == 200,
            f"available={drift.json().get('available')}" if drift.status_code == 200 else "",
        )

        metrics = get("/metrics")
        _check(
            results,
            "GET /metrics",
            metrics.status_code == 200 and "api_requests_total" in metrics.text,
        )

        graphql = requests.post(
            f"{base}/graphql",
            json={"query": "{ modelInfo { name version } }"},
            timeout=args.timeout,
        )
        _check(
            results,
            "POST /graphql",
            graphql.status_code == 200 and "errors" not in graphql.json(),
        )

    except requests.RequestException as exc:
        _check(results, "connectivity", False, str(exc))

    try:
        recent = get("/predictions?limit=1")
        if recent.status_code == 200 and recent.json().get("items"):
            print("most recent stored prediction:")
            print(json.dumps(recent.json()["items"][0], indent=2, default=str))
    except requests.RequestException:
        pass

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
