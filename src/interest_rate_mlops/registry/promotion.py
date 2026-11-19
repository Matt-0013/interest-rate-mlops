"""Model promotion rules.

A newly trained model is **not** promoted automatically. It becomes the
champion only when every gate passes:

1. absolute quality - held-out RMSE within the configured usefulness ceiling;
2. error sanity      - MAPE (where computable) within the configured ceiling;
3. stability         - walk-forward fold RMSE variation is not extreme;
4. regression guard  - it beats the incumbent champion by the configured margin
   (or, when no champion exists, the bootstrap rule applies).

Every gate result is recorded with a human-readable reason, which is what makes
a rejection auditable later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from interest_rate_mlops.config import PromotionConfig
from interest_rate_mlops.exceptions import PromotionError
from interest_rate_mlops.logging_utils import get_logger
from interest_rate_mlops.registry.mlflow_registry import ModelRegistry

logger = get_logger(__name__)


@dataclass(frozen=True)
class GateResult:
    """Outcome of one promotion gate."""

    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    """Final approve/reject decision with full audit trail."""

    approved: bool
    candidate_version: str | None
    champion_version: str | None
    candidate_metrics: dict[str, float | None]
    champion_metrics: dict[str, float | None]
    improvement_pct: float | None
    gates: list[GateResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        state = "APPROVED" if self.approved else "REJECTED"
        return f"{state}: " + "; ".join(
            f"{gate.name}={'pass' if gate.passed else 'fail'}" for gate in self.gates
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "candidate_version": self.candidate_version,
            "champion_version": self.champion_version,
            "candidate_metrics": self.candidate_metrics,
            "champion_metrics": self.champion_metrics,
            "improvement_pct": self.improvement_pct,
            "gates": [gate.as_dict() for gate in self.gates],
            "reasons": list(self.reasons),
            "summary": self.summary,
        }


def _metric(metrics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int | float):
            return float(value)
    return None


def evaluate_promotion(
    candidate_metrics: Mapping[str, Any],
    champion_metrics: Mapping[str, Any] | None,
    cfg: PromotionConfig,
    *,
    candidate_version: str | None = None,
    champion_version: str | None = None,
) -> PromotionDecision:
    """Apply every promotion gate to a candidate.

    Args:
        candidate_metrics: held-out/CV metrics of the candidate (``test_rmse``,
            ``test_mape``, ``cv_rmse_std_ratio``).
        champion_metrics: metrics of the incumbent, or ``None`` when there is no
            champion yet.
        cfg: thresholds from ``configs/model.yaml``.
    """
    gates: list[GateResult] = []
    reasons: list[str] = []

    test_rmse = _metric(candidate_metrics, "test_rmse", "rmse")
    test_mape = _metric(candidate_metrics, "test_mape", "mape")
    cv_ratio = _metric(candidate_metrics, "cv_rmse_std_ratio")
    if cv_ratio is None:
        cv_std = _metric(candidate_metrics, "cv_rmse_std")
        cv_mean = _metric(candidate_metrics, "cv_rmse_mean")
        if cv_std is not None and cv_mean:
            cv_ratio = cv_std / cv_mean

    # Gate 1 - absolute quality.
    quality_ok = test_rmse is not None and test_rmse <= cfg.max_test_rmse
    gates.append(
        GateResult(
            name="absolute_quality",
            passed=quality_ok,
            detail=(
                f"test RMSE {test_rmse:.4f} <= {cfg.max_test_rmse}"
                if test_rmse is not None
                else "test RMSE missing"
            ),
        )
    )
    if not quality_ok:
        reasons.append(
            f"held-out RMSE {test_rmse if test_rmse is None else round(test_rmse, 4)} "
            f"exceeds the usefulness ceiling {cfg.max_test_rmse}"
        )

    # Gate 2 - MAPE sanity, only when MAPE is computable.
    if test_mape is None:
        gates.append(
            GateResult(
                name="mape_sanity",
                passed=True,
                detail="MAPE not computable (no |actual| >= 0.5 pp rows); gate skipped",
            )
        )
    else:
        mape_ok = test_mape <= cfg.max_mape
        gates.append(
            GateResult(
                name="mape_sanity",
                passed=mape_ok,
                detail=f"test MAPE {test_mape:.2f}% <= {cfg.max_mape}%",
            )
        )
        if not mape_ok:
            reasons.append(f"MAPE {round(test_mape, 2)}% exceeds {cfg.max_mape}%")

    # Gate 3 - stability across walk-forward folds.
    stability_ok = cv_ratio is not None and cv_ratio <= cfg.max_cv_rmse_std_ratio
    gates.append(
        GateResult(
            name="cv_stability",
            passed=stability_ok,
            detail=(
                f"fold RMSE std/mean {cv_ratio:.4f} <= {cfg.max_cv_rmse_std_ratio}"
                if cv_ratio is not None
                else "CV dispersion missing"
            ),
        )
    )
    if not stability_ok:
        reasons.append("cross-validation error is too unstable to promote")

    # Gate 4 - must beat the champion.
    champion_rmse = _metric(champion_metrics or {}, "test_rmse", "rmse")
    improvement: float | None = None
    if champion_metrics is None or champion_rmse is None:
        passed = cfg.bootstrap_when_no_champion
        detail = (
            "no incumbent champion; bootstrap promotion allowed"
            if passed
            else "no incumbent champion and bootstrap promotion disabled"
        )
        if not passed:
            reasons.append("no champion available and bootstrap promotion is disabled")
    else:
        improvement = (champion_rmse - (test_rmse or 0.0)) / champion_rmse * 100.0
        passed = improvement >= cfg.min_rmse_improvement_pct
        detail = (
            f"RMSE improvement {improvement:.2f}% "
            f"{'>= ' if passed else '< '}{cfg.min_rmse_improvement_pct}% vs champion "
            f"(champion {champion_rmse:.4f}, candidate "
            f"{test_rmse if test_rmse is None else round(test_rmse, 4)})"
        )
        if not passed:
            reasons.append(
                f"candidate does not beat the champion by {cfg.min_rmse_improvement_pct}% "
                f"(improvement {round(improvement, 2)}%)"
            )
    gates.append(GateResult(name="beats_champion", passed=passed, detail=detail))

    approved = all(gate.passed for gate in gates)
    if approved:
        reasons.append("candidate satisfies every promotion gate")

    decision = PromotionDecision(
        approved=approved,
        candidate_version=candidate_version,
        champion_version=champion_version,
        candidate_metrics=dict(candidate_metrics),
        champion_metrics=dict(champion_metrics or {}),
        improvement_pct=improvement,
        gates=gates,
        reasons=reasons,
    )
    logger.info(
        "promotion decision",
        extra={
            "event": "promotion_decision",
            "model_version": candidate_version,
            "status_code": 200 if approved else 409,
        },
    )
    return decision


def apply_promotion(
    registry: ModelRegistry,
    decision: PromotionDecision,
    champion_alias: str,
) -> None:
    """Move the champion alias and tag the version with the audit trail.

    Raises:
        PromotionError: when the decision is a rejection or the version is
            unknown. Rejections are never silently applied.
    """
    if not decision.approved:
        raise PromotionError(f"refusing to promote a rejected candidate ({decision.summary})")
    if decision.candidate_version is None:
        raise PromotionError("candidate has no registered version")

    registry.set_alias(champion_alias, decision.candidate_version)
    registry.tag_version(
        decision.candidate_version,
        "promotion_reason",
        decision.summary[:500],
    )
    if decision.improvement_pct is not None:
        registry.tag_version(
            decision.candidate_version,
            "rmse_improvement_pct_vs_champion",
            f"{decision.improvement_pct:.4f}",
        )
    logger.info(
        "promoted candidate to champion",
        extra={
            "event": "model_promoted",
            "model_name": registry.model_name,
            "model_version": decision.candidate_version,
        },
    )
