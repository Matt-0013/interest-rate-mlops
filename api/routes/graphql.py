"""GraphQL surface.

The REST endpoints are the primary contract; GraphQL exposes the same operations
for clients that prefer a typed single endpoint. Both call the identical service
functions, so validation, persistence and metrics cannot drift apart.
"""

from __future__ import annotations

from typing import Any

import strawberry
from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON
from strawberry.types import Info

from api.schemas.prediction import HistoryObservation, PredictionRequest
from api.services import perform_prediction, read_drift_status


@strawberry.input
class ObservationInput:
    date: str
    value: float


@strawberry.input
class PredictInput:
    features: JSON | None = None
    history: list[ObservationInput] | None = None


@strawberry.type
class ModelInfo:
    name: str
    version: str
    alias: str
    target: str
    unit: str
    horizon: int
    feature_names: list[str]
    lookback_observations: int
    dataset_version: str | None
    metrics: JSON
    loaded_at: str


@strawberry.type
class PredictionResult:
    request_id: str
    prediction: float
    unit: str
    target: str
    model_name: str
    model_version: str
    input_mode: str
    origin_date: str | None


@strawberry.type
class DriftStatus:
    available: bool
    status: str
    data_drift: bool | None
    drift_share: float | None
    drift_status: str | None
    performance_status: str | None
    retrain_recommended: bool | None
    alerts: list[str]
    generated_at: str | None


def _request(info: Info) -> Request:
    return info.context["request"]


def _session(request: Request) -> Session:
    database = request.app.state.db
    if database is None or database.session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database unavailable: predictions cannot be persisted",
        )
    return database.session_factory()


@strawberry.type
class Query:
    @strawberry.field
    def model_info(self, info: Info) -> ModelInfo:
        handle = _request(info).app.state.model_service.get()
        return ModelInfo(
            name=handle.name,
            version=handle.version,
            alias=handle.alias,
            target=handle.target,
            unit=handle.unit,
            horizon=handle.horizon,
            feature_names=list(handle.feature_names),
            lookback_observations=handle.lookback_observations,
            dataset_version=handle.dataset_version,
            metrics=dict(handle.metrics),
            loaded_at=handle.loaded_at.isoformat(),
        )

    @strawberry.field
    def drift_status(self, info: Info) -> DriftStatus:
        request = _request(info)
        database = request.app.state.db
        session: Session | None = None
        if database is not None and database.session_factory is not None:
            try:
                session = database.session_factory()
            except Exception:  # noqa: BLE001 - file fallback
                session = None
        try:
            payload = read_drift_status(session, request.app.state.settings)
        finally:
            if session is not None:
                session.close()
        return DriftStatus(
            available=payload.available,
            status=payload.status,
            data_drift=payload.data_drift,
            drift_share=payload.drift_share,
            drift_status=payload.drift_status,
            performance_status=payload.performance_status,
            retrain_recommended=payload.retrain_recommended,
            alerts=list(payload.alerts),
            generated_at=payload.generated_at,
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    def predict(self, info: Info, payload: PredictInput) -> PredictionResult:
        request = _request(info)
        model_payload = PredictionRequest(
            features=dict(payload.features) if payload.features else None,
            history=[
                HistoryObservation(date=observation.date, value=observation.value)
                for observation in (payload.history or [])
            ]
            or None,
        )
        handle = request.app.state.model_service.get()
        session = _session(request)
        try:
            result = perform_prediction(
                model_payload,
                handle,
                session,
                request.app.state.settings,
                source="graphql",
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return PredictionResult(
            request_id=result.request_id,
            prediction=result.prediction,
            unit=result.unit,
            target=result.target,
            model_name=result.model_name,
            model_version=result.model_version,
            input_mode=result.input_mode,
            origin_date=result.origin_date,
        )


schema = strawberry.Schema(query=Query, mutation=Mutation)


async def _context(request: Request) -> dict[str, Any]:
    return {"request": request}


router = GraphQLRouter(schema, context_getter=_context, graphiql=True)
