"""
tests/unit/test_gout_rules.py
------------------------------
Unit tests for GoutRules.evaluate().

Coverage target: >= 90% branch coverage on rules/gout.py
"""

from __future__ import annotations
from typing import Any

import pytest

from urolens_ai.smart_diagnosis.rules.gout import GoutRules, RuleResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOUT_CONFIG: dict[str, Any] = {
    "particles": {
        "crystals": {
            "weight": 1.0,
            "normal_range_max": 5,
            "display_name": "Crystals (Uric Acid)",
        }
    },
    "thresholds": {
        "low_max_score": 3.0,
        "high_min_score": 10.0,
    },
    "evidence_min_count": 2,
}


@pytest.fixture
def gout_rules() -> GoutRules:
    return GoutRules(config=GOUT_CONFIG)


# ---------------------------------------------------------------------------
# All particles at zero
# ---------------------------------------------------------------------------


class TestGoutRulesZeroCounts:
    def test_all_zero_score_is_zero(self, gout_rules: GoutRules) -> None:
        """All particles at zero count must produce weighted_score = 0.0."""
        result = gout_rules.evaluate({"crystals": 0})
        assert result.weighted_score == 0.0

    def test_all_zero_no_matched_particles(self, gout_rules: GoutRules) -> None:
        """All particles at zero count must produce empty matched_particles."""
        result = gout_rules.evaluate({"crystals": 0})
        assert result.matched_particles == []

    def test_empty_classification_score_is_zero(self, gout_rules: GoutRules) -> None:
        """Empty classification dict must produce weighted_score = 0.0."""
        result = gout_rules.evaluate({})
        assert result.weighted_score == 0.0


# ---------------------------------------------------------------------------
# Within normal range
# ---------------------------------------------------------------------------


class TestGoutRulesWithinNormalRange:
    def test_at_normal_max_score_is_zero(self, gout_rules: GoutRules) -> None:
        """Crystals at exactly normal_range_max must not contribute to score."""
        result = gout_rules.evaluate({"crystals": 5})
        assert result.weighted_score == 0.0

    def test_at_normal_max_no_matched_particles(self, gout_rules: GoutRules) -> None:
        """Crystals at exactly normal_range_max must not appear in matched_particles."""
        result = gout_rules.evaluate({"crystals": 5})
        assert result.matched_particles == []

    def test_below_normal_max_score_is_zero(self, gout_rules: GoutRules) -> None:
        """Crystals below normal_range_max must produce score = 0.0."""
        result = gout_rules.evaluate({"crystals": 3})
        assert result.weighted_score == 0.0


# ---------------------------------------------------------------------------
# Above normal range
# ---------------------------------------------------------------------------


class TestGoutRulesAboveNormalRange:
    def test_one_above_normal_score_correct(self, gout_rules: GoutRules) -> None:
        """Crystals at 6 (1 above normal max 5, weight 1.0) must produce score 1.0."""
        result = gout_rules.evaluate({"crystals": 6})
        assert result.weighted_score == pytest.approx(1.0) # type: ignore[no-untyped-call]

    def test_far_above_normal_score_correct(self, gout_rules: GoutRules) -> None:
        """Crystals at 15 (10 above normal max 5, weight 1.0) must produce score 10.0."""
        result = gout_rules.evaluate({"crystals": 15})
        assert result.weighted_score == pytest.approx(10.0) # type: ignore[no-untyped-call]

    def test_matched_particle_fields_correct(self, gout_rules: GoutRules) -> None:
        """Matched particle must have correct name, count, contribution."""
        result = gout_rules.evaluate({"crystals": 12})
        assert len(result.matched_particles) == 1
        particle = result.matched_particles[0]
        assert particle.name == "crystals"
        assert particle.detected_count == 12
        assert particle.normal_range_max == 5
        assert particle.weight == pytest.approx(1.0) # type: ignore[no-untyped-call]
        assert particle.contribution == pytest.approx(7.0) # type: ignore[no-untyped-call]

    def test_contribution_formula_correct(self, gout_rules: GoutRules) -> None:
        """contribution must equal (detected_count - normal_range_max) * weight."""
        result = gout_rules.evaluate({"crystals": 8})
        particle = result.matched_particles[0]
        expected = (8 - 5) * 1.0
        assert particle.contribution == pytest.approx(expected) # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Missing particle key
# ---------------------------------------------------------------------------


class TestGoutRulesMissingKey:
    def test_missing_particle_treated_as_zero(self, gout_rules: GoutRules) -> None:
        """Particle absent from classification must be treated as count = 0."""
        result = gout_rules.evaluate({})
        assert result.weighted_score == 0.0
        assert result.matched_particles == []

    def test_unknown_key_ignored(self, gout_rules: GoutRules) -> None:
        """Unknown particle keys in classification must be ignored."""
        result = gout_rules.evaluate({"unknown_particle": 999})
        assert result.weighted_score == 0.0


# ---------------------------------------------------------------------------
# RuleResult fields
# ---------------------------------------------------------------------------


class TestGoutRuleResultFields:
    def test_condition_is_gout(self, gout_rules: GoutRules) -> None:
        """RuleResult.condition must always be 'gout'."""
        result = gout_rules.evaluate({})
        assert result.condition == "gout"

    def test_returns_rule_result_instance(self, gout_rules: GoutRules) -> None:
        """evaluate() must return a RuleResult instance."""
        result = gout_rules.evaluate({})
        assert isinstance(result, RuleResult)

    def test_weighted_score_is_float(self, gout_rules: GoutRules) -> None:
        """weighted_score must always be a float."""
        result = gout_rules.evaluate({"crystals": 10})
        assert isinstance(result.weighted_score, float)