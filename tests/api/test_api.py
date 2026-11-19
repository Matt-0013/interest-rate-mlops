"""API behaviour: happy paths, validation, persistence and failure modes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.dependencies.database import Database
from api.main import create_app
from interest_rate_mlops.config import load_pipeline_config

FEATURES = load_pipeline_config().features


# ---------------------------------------------------------------------------
# Health & model info
# ---------------------------------------------------------------------------
def test_health_reports_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["checks"] == {"database": True, "model": True}
    assert {dependency["name"] for dependency in body["dependencies"]} == {
        "database",
        "model_registry",
    }


def test_ready_probe(client: TestClient):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_model_info_returns_champion_contract(client: TestClient):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "interest_rate_forecaster"
    assert body["version"] == "7"
    assert body["alias"] == "champion"
    assert body["feature_names"] == FEATURES.names()
    assert body["lookback_observations"] == FEATURES.lookback
    assert body["unit"] == "percent per annum"
    assert body["metrics"]["test_rmse"] == pytest.approx(0.1585)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def test_predict_with_features(client: TestClient, features_payload):
    response = client.post("/predict", json={"features": features_payload})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == pytest.approx(2.5)
    assert body["input_mode"] == "features"
    assert body["model_version"] == "7"
    assert body["request_id"]
    assert body["latency_ms"] >= 0


def test_predict_with_history_computes_origin(client: TestClient, history_payload):
    response = client.post("/predict", json={"history": history_payload})
    assert response.status_code == 200
    body = response.json()
    assert body["input_mode"] == "history"
    assert body["origin_date"] == history_payload[-1]["date"]
    assert body["prediction"] == pytest.approx(2.5)


def test_predict_persists_prediction(client: TestClient, features_payload):
    client.post("/predict", json={"features": features_payload})
    client.post("/predict", json={"features": features_payload})

    response = client.get("/predictions")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["model_version"] == "7"
    assert body["items"][0]["features"] == pytest.approx(features_payload)


def test_predictions_pagination(client: TestClient, features_payload):
    for _ in range(3):
        client.post("/predict", json={"features": features_payload})

    response = client.get("/predictions", params={"limit": 2, "offset": 1})
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_predict_requires_exactly_one_input_mode(
    client: TestClient, features_payload, history_payload
):
    both = client.post("/predict", json={"features": features_payload, "history": history_payload})
    assert both.status_code == 422

    neither = client.post("/predict", json={})
    assert neither.status_code == 422


def test_predict_rejects_unknown_field(client: TestClient, features_payload):
    response = client.post("/predict", json={"features": features_payload, "model": "x"})
    assert response.status_code == 422


def test_predict_rejects_missing_feature(client: TestClient, features_payload):
    payload = dict(features_payload)
    payload.pop(FEATURES.names()[0])
    response = client.post("/predict", json={"features": payload})
    assert response.status_code == 422
    assert "missing required features" in response.json()["detail"]
    assert response.json()["failures"] == ["missing_features"]


def test_predict_rejects_unexpected_feature(client: TestClient, features_payload):
    payload = dict(features_payload)
    payload["made_up_feature"] = 1.0
    response = client.post("/predict", json={"features": payload})
    assert response.status_code == 422
    assert response.json()["failures"] == ["unexpected_features"]


def test_predict_rejects_non_numeric_and_out_of_range(client: TestClient, features_payload):
    non_numeric = dict(features_payload)
    non_numeric[FEATURES.names()[0]] = "abc"
    assert client.post("/predict", json={"features": non_numeric}).status_code == 422

    out_of_range = dict(features_payload)
    out_of_range["lag_1"] = 99.0
    response = client.post("/predict", json={"features": out_of_range})
    assert response.status_code == 422
    assert response.json()["failures"] == ["feature_out_of_range"]


def test_predict_rejects_invalid_history(client: TestClient, history_payload):
    gapped = [history_payload[0], history_payload[-1]]
    response = client.post("/predict", json={"history": gapped})
    assert response.status_code == 422

    bad_date = [{"date": "2016-01-15", "value": 1.0}, {"date": "2016-02-01", "value": 1.1}]
    assert client.post("/predict", json={"history": bad_date}).status_code == 422


def test_predict_returns_503_without_champion(client: TestClient, model_service, features_payload):
    model_service.available = False
    response = client.post("/predict", json={"features": features_payload})
    assert response.status_code == 503
    assert client.get("/model/info").status_code == 503


def test_predict_returns_503_without_database(api_settings, model_service, features_payload):
    settings = api_settings.model_copy(
        update={"database_url": "sqlite+pysqlite:////nonexistent_root_dir/mlops.sqlite"}
    )
    app = create_app(settings=settings, database=Database(settings), model_service=model_service)
    with TestClient(app) as failing_client:
        response = failing_client.post("/predict", json={"features": features_payload})
        assert response.status_code == 503
        health = failing_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert failing_client.get("/health/ready").status_code == 503


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
def test_metrics_endpoint_exposes_prometheus_text(client: TestClient):
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert response.headers["content-type"].startswith("text/plain")


def test_drift_status_without_monitoring_data(client: TestClient, monkeypatch):
    monkeypatch.setattr("api.services.read_json", lambda *args, **kwargs: None)
    response = client.get("/drift/status")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "make monitor" in body["notes"][0]


def test_drift_status_falls_back_to_file(client: TestClient, monkeypatch):
    document = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "status": "warning",
        "data_drift": True,
        "drift_share": 0.4,
        "drift_status": "warning",
        "performance_status": "ok",
        "retrain_recommended": False,
        "alerts": ["data drift warning"],
        "windows": {"reference_rows": 100, "current_rows": 30},
        "drift": {"report_path": "reports/monitoring/x.html"},
        "notes": ["drift is not accuracy"],
    }
    monkeypatch.setattr("api.services.read_json", lambda *args, **kwargs: document)
    response = client.get("/drift/status")
    body = response.json()
    assert body["available"] is True
    assert body["source"] == "file"
    assert body["drift_status"] == "warning"
    assert body["performance_status"] == "ok"


# ---------------------------------------------------------------------------
# GraphQL
# ---------------------------------------------------------------------------
def test_graphql_model_info(client: TestClient):
    query = "{ modelInfo { name version alias featureNames } }"
    response = client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    payload = response.json()
    assert "errors" not in payload, payload
    assert payload["data"]["modelInfo"]["version"] == "7"
    assert payload["data"]["modelInfo"]["featureNames"] == FEATURES.names()


def test_graphql_predict(client: TestClient, features_payload):
    query = """
        mutation Predict($payload: PredictInput!) {
          predict(payload: $payload) { prediction modelVersion inputMode }
        }
    """
    response = client.post(
        "/graphql", json={"query": query, "variables": {"payload": {"features": features_payload}}}
    )
    assert response.status_code == 200
    payload = response.json()
    assert "errors" not in payload, payload
    data = payload["data"]["predict"]
    assert data["prediction"] == pytest.approx(2.5)
    assert data["modelVersion"] == "7"
    assert data["inputMode"] == "features"
