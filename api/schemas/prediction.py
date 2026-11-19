"""Prediction schemas.

Two input modes are accepted, exactly one per request:

* ``features`` - the model's engineered feature contract (fast path for ML
  consumers that already compute features);
* ``history``  - a contiguous monthly series, converted with the same feature
  code used in training (removes train/serve skew).
"""

from __future__ import annotations

from datetime import date as DateType
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HistoryObservation(BaseModel):
    """One monthly observation of the target series."""

    model_config = ConfigDict(extra="forbid")

    date: DateType = Field(description="Month-start observation date (YYYY-MM-01).")
    value: float = Field(description="Observed value in the target's unit (percent).")


class PredictionRequest(BaseModel):
    """Request body for ``POST /predict``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "history": [
                        {"date": "2016-08-01", "value": 0.40},
                        {"date": "2016-09-01", "value": 0.41},
                        {"date": "2016-10-01", "value": 0.42},
                        {"date": "2016-11-01", "value": 0.41},
                        {"date": "2016-12-01", "value": 0.42},
                        {"date": "2017-01-01", "value": 0.65},
                    ]
                }
            ]
        },
    )

    features: dict[str, float] | None = Field(
        default=None,
        description="Engineered features (see GET /model/info for the exact contract).",
    )
    history: list[HistoryObservation] | None = Field(
        default=None,
        description="Contiguous monthly observations; at least the model lookback is required.",
    )

    @model_validator(mode="after")
    def _exactly_one_mode(self) -> PredictionRequest:
        if (self.features is None) == (self.history is None):
            raise ValueError("provide exactly one of 'features' or 'history'")
        return self


class PredictionResponse(BaseModel):
    """Result of one forecast."""

    request_id: str
    prediction: float
    unit: str
    target: str
    model_name: str
    model_version: str
    model_alias: str
    input_mode: Literal["features", "history"]
    origin_date: str | None = None
    features: dict[str, float]
    latency_ms: float
    created_at: datetime


class PredictionRecordOut(BaseModel):
    """Stored prediction returned by ``GET /predictions``."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    request_id: str
    model_name: str
    model_version: str
    prediction: float
    unit: str
    origin_date: str | None
    features: dict[str, Any]
    latency_ms: float | None
    created_at: datetime


class PredictionListResponse(BaseModel):
    """Paginated prediction history."""

    items: list[PredictionRecordOut]
    total: int
    limit: int
    offset: int
