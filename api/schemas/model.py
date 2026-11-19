"""Model registry information exposed by the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Currently approved (champion) model."""

    name: str
    version: str
    alias: str
    run_id: str | None = None
    dataset_version: str | None = None
    target: str
    unit: str
    horizon: int
    feature_names: list[str] = Field(default_factory=list)
    lookback_observations: int
    metrics: dict[str, float] = Field(default_factory=dict)
    loaded_at: datetime
    cache_ttl_seconds: int
    tracking_uri: str
    notes: list[str] = Field(
        default_factory=list,
        description="Caveats about the model contract or serving setup.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)
