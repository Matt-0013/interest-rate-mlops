"""HTTP route modules."""

from api.routes import drift, health, metrics, model, predict, predictions

ROUTERS = [
    health.router,
    model.router,
    predict.router,
    predictions.router,
    metrics.router,
    drift.router,
]

__all__ = ["ROUTERS"]
