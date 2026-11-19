"""Model evaluation and promotion pipeline (Airflow DAG 3, ``make promote``).

Compares the registered candidate against the current champion, applies every
gate, and only then moves the ``champion`` alias. Rejections are recorded with
reasons and leave the incumbent untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from interest_rate_mlops.config import PipelineConfig, Settings, get_settings, load_pipeline_config
from interest_rate_mlops.database.repositories import (
    record_deployment,
    record_pipeline_execution,
    upsert_model_metadata,
)
from interest_rate_mlops.exceptions import MLOpsError
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.pipelines.common import (
    Timer,
    optional_session,
    record_pipeline_state,
    utc_stamp,
)
from interest_rate_mlops.registry import (
    ModelRegistry,
    PromotionDecision,
    RegisteredModelInfo,
    apply_promotion,
    evaluate_promotion,
)
from interest_rate_mlops.storage import write_json

logger = get_logger(__name__)


@dataclass(frozen=True)
class PromotionOutcome:
    """Result of one promotion evaluation."""

    decision: PromotionDecision
    promoted: bool
    candidate: RegisteredModelInfo | None = None
    champion_before: RegisteredModelInfo | None = None
    report_path: str = ""
    db_persisted: bool = False
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "candidate_version": self.candidate.version if self.candidate else None,
            "champion_before": self.champion_before.as_dict() if self.champion_before else None,
            "decision": self.decision.as_dict(),
            "report_path": self.report_path,
            "db_persisted": self.db_persisted,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def _resolve_candidate(
    registry: ModelRegistry, cfg: PipelineConfig, challenger_version: str | None
) -> RegisteredModelInfo:
    if challenger_version:
        return registry.get_version(challenger_version)

    challenger = registry.get_version_by_alias(cfg.mlflow.challenger_alias)
    if challenger is not None:
        return challenger

    versions = registry.list_versions()
    if not versions:
        raise MLOpsError(
            f"no registered versions of {registry.model_name!r}; run the training "
            "pipeline before promotion"
        )
    return versions[-1]


def run_promotion(
    *,
    challenger_version: str | None = None,
    settings: Settings | None = None,
    pipeline_cfg: PipelineConfig | None = None,
) -> PromotionOutcome:
    """Evaluate the candidate against the champion and promote when eligible."""
    settings = settings or get_settings()
    pipeline_cfg = pipeline_cfg or load_pipeline_config()
    registry = ModelRegistry(
        settings.resolved_tracking_uri, pipeline_cfg.mlflow.registered_model_name
    )
    champion_alias = pipeline_cfg.mlflow.champion_alias

    with Timer() as timer:
        candidate = _resolve_candidate(registry, pipeline_cfg, challenger_version)
        champion = registry.get_version_by_alias(champion_alias)

        candidate_metrics = registry.model_metrics(candidate.version)
        champion_metrics = (
            registry.model_metrics(champion.version)
            if champion is not None and champion.version != candidate.version
            else None
        )

        decision = evaluate_promotion(
            candidate_metrics,
            champion_metrics,
            pipeline_cfg.promotion,
            candidate_version=candidate.version,
            champion_version=champion.version if champion else None,
        )

        promoted = False
        if decision.approved:
            apply_promotion(registry, decision, champion_alias)
            promoted = True
        else:
            logger.info(
                "candidate rejected; champion unchanged",
                extra={
                    "event": "promotion_rejected",
                    "model_version": candidate.version,
                    "status_code": 409,
                },
            )

        report_dir = Path(settings.reports_path) / "promotion"
        report_path = write_json(
            report_dir / f"promotion_{utc_stamp()}.json",
            {
                **decision.as_dict(),
                "candidate": candidate.as_dict(),
                "champion_before": champion.as_dict() if champion else None,
                "promoted": promoted,
            },
        )
        write_json(report_dir / "latest_promotion.json", decision.as_dict())

        db_persisted = False
        with optional_session(settings) as session:
            if session is not None:
                record_deployment(
                    session,
                    model_name=pipeline_cfg.mlflow.registered_model_name,
                    to_version=candidate.version,
                    from_version=champion.version if champion else None,
                    action="promotion" if promoted else "rejection",
                    reason=decision.summary,
                    decision=decision.as_dict(),
                )
                upsert_model_metadata(
                    session,
                    model_name=pipeline_cfg.mlflow.registered_model_name,
                    version=candidate.version,
                    alias=champion_alias if promoted else pipeline_cfg.mlflow.challenger_alias,
                    stage="champion" if promoted else "challenger",
                    metrics=candidate_metrics,
                    run_id=candidate.run_id,
                    dataset_version=candidate.tags.get("dataset_version"),
                    is_active=promoted,
                )
                record_pipeline_execution(
                    session,
                    pipeline="promotion",
                    status="succeeded" if promoted else "skipped",
                    duration_seconds=timer.seconds,
                    model_version=candidate.version,
                    details={"approved": promoted, "gates": [g.name for g in decision.gates]},
                )
                db_persisted = True

        record_pipeline_state(
            settings,
            pipeline="promotion",
            status="succeeded" if promoted else "skipped",
            duration_seconds=timer.seconds,
            details={
                "candidate_version": candidate.version,
                "approved": promoted,
                "improvement_pct": decision.improvement_pct,
            },
        )

    return replace(
        PromotionOutcome(
            decision=decision,
            promoted=promoted,
            candidate=candidate,
            champion_before=champion,
            report_path=str(report_path),
            duration_seconds=timer.seconds,
        ),
        db_persisted=db_persisted,
    )
