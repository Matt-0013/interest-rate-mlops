"""Promotion rules: quality, sanity, stability and regression gates."""

from __future__ import annotations

import pytest

from interest_rate_mlops.config import PromotionConfig, load_pipeline_config
from interest_rate_mlops.exceptions import PromotionError
from interest_rate_mlops.registry.promotion import (
    PromotionDecision,
    apply_promotion,
    evaluate_promotion,
)

CFG = PromotionConfig(
    max_test_rmse=0.75,
    max_mape=25.0,
    max_cv_rmse_std_ratio=1.0,
    min_rmse_improvement_pct=2.0,
    bootstrap_when_no_champion=True,
)

GOOD = {
    "test_rmse": 0.15,
    "test_mae": 0.11,
    "test_mape": 6.0,
    "cv_rmse_std_ratio": 0.8,
}


def test_bootstrap_promotion_when_all_gates_pass():
    decision = evaluate_promotion(GOOD, None, CFG, candidate_version="1")
    assert decision.approved
    assert decision.improvement_pct is None
    assert all(gate.passed for gate in decision.gates)
    assert "bootstrap" in decision.gates[-1].detail


def test_bootstrap_can_be_disabled():
    config = CFG.model_copy(update={"bootstrap_when_no_champion": False})
    decision = evaluate_promotion(GOOD, None, config)
    assert not decision.approved
    assert decision.gates[-1].name == "beats_champion"


def test_poor_absolute_quality_is_rejected():
    decision = evaluate_promotion({**GOOD, "test_rmse": 0.9}, None, CFG)
    assert not decision.approved
    assert decision.gates[0].name == "absolute_quality"
    assert any("usefulness ceiling" in reason for reason in decision.reasons)


def test_mape_gate_is_skipped_when_not_computable():
    candidate = {**GOOD, "test_mape": None}
    decision = evaluate_promotion(candidate, None, CFG)
    assert decision.approved
    mape_gate = next(gate for gate in decision.gates if gate.name == "mape_sanity")
    assert mape_gate.passed
    assert "not computable" in mape_gate.detail


def test_unstable_cv_is_rejected():
    decision = evaluate_promotion({**GOOD, "cv_rmse_std_ratio": 1.6}, None, CFG)
    assert not decision.approved
    assert not next(g for g in decision.gates if g.name == "cv_stability").passed


def test_candidate_must_beat_incumbent_by_margin():
    champion = {"test_rmse": 0.15}
    barely_better = evaluate_promotion({**GOOD, "test_rmse": 0.148}, champion, CFG)
    assert not barely_better.approved
    assert barely_better.improvement_pct == pytest.approx(1.33, abs=0.01)

    clearly_better = evaluate_promotion({**GOOD, "test_rmse": 0.12}, champion, CFG)
    assert clearly_better.approved
    assert clearly_better.improvement_pct == pytest.approx(20.0)


def test_missing_metrics_are_rejected_not_assumed():
    decision = evaluate_promotion({"test_mae": 0.1}, None, CFG)
    assert not decision.approved
    assert "test RMSE missing" in decision.gates[0].detail


class FakeRegistry:
    model_name = "forecaster"

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.tags: dict[tuple[str, str], str] = {}

    def set_alias(self, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def tag_version(self, version: str, key: str, value: str) -> None:
        self.tags[(version, key)] = value


def test_apply_promotion_moves_alias_and_tags_version():
    registry = FakeRegistry()
    decision = PromotionDecision(
        approved=True,
        candidate_version="3",
        champion_version="2",
        candidate_metrics=GOOD,
        champion_metrics={"test_rmse": 0.2},
        improvement_pct=25.0,
        gates=[],
        reasons=["candidate satisfies every promotion gate"],
    )
    apply_promotion(registry, decision, "champion")

    assert registry.aliases["champion"] == "3"
    assert registry.tags[("3", "promotion_reason")].startswith("APPROVED")
    assert registry.tags[("3", "rmse_improvement_pct_vs_champion")] == "25.0000"


def test_apply_promotion_refuses_rejected_candidate():
    registry = FakeRegistry()
    decision = PromotionDecision(
        approved=False,
        candidate_version="3",
        champion_version="2",
        candidate_metrics=GOOD,
        champion_metrics={"test_rmse": 0.1},
        improvement_pct=-50.0,
    )
    with pytest.raises(PromotionError, match="refusing to promote"):
        apply_promotion(registry, decision, "champion")
    assert registry.aliases == {}


def test_config_thresholds_are_positive():
    config = load_pipeline_config().promotion
    assert config.max_test_rmse > 0
    assert config.min_rmse_improvement_pct >= 0
    assert config.max_cv_rmse_std_ratio > 0
