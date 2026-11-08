"""
tests/unit/test_nephrolithiasis_rules.py
-----------------------------------------
Unit tests for NephrolithiasisRules.evaluate().

Coverage target: >= 90% branch coverage on rules/nephrolithiasis.py
"""

from __future__ import annotations
from typing import Any

import pytest

from urolens_ai.smart_diagnosis.rules.gout import RuleResult
from urolens_ai.smart_diagnosis.rules.nephrolithiasis import NephrolithiasisRules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NEPHRO_CONFIG: dict[str, Any] = {
    "particles": {
        "crystals": {
            "weight": 0.3,
            "normal_range_max": 5,
            "display_name": "Crystals (Stone-forming)",
        },
        "erythrocytes": {
            "weight": 0.1,
            "normal_range_max": 3,
            "display_name": "Erythrocytes (Red Blood Cells)",
        },
    },
    "thresholds": {
        "low_max_score": 3.0,
        "high_min_score": 15.0,
    },
    "evidence_min_count": 2,
}


@pytest.fixture
def nephro_rules() -> NephrolithiasisRules:
    return NephrolithiasisRules(config=NEPHRO_CONFIG)


# ---------------------------------------------------------------------------
# All within normal range
# ---------------------------------------------------------------------------


class TestNephroRulesWithinNormalRange:
    def test_all_zero_score_is_zero(self, nephro_rules: NephrolithiasisRules) -> None:
        """All particles at zero must produce weighted_score = 0.0."""
        result = nephro_rules.evaluate({"crystals": 0, "erythrocytes": 0})
        assert result.weighted_score == 0.0

    def test_crystals_at_normal_max_score_zero(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Crystals at exactly normal_range_max (5) must not contribute."""
        result = nephro_rules.evaluate({"crystals": 5})
        assert result.weighted_score == 0.0

    def test_erythrocytes_at_normal_max_score_zero(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Erythrocytes at exactly normal_range_max (3) must not contribute."""
        result = nephro_rules.evaluate({"erythrocytes": 3})
        assert result.weighted_score == 0.0

    def test_empty_classification_score_zero(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Empty classification must produce score = 0.0."""
        result = nephro_rules.evaluate({})
        assert result.weighted_score == 0.0


# ---------------------------------------------------------------------------
# Crystals above normal range
# ---------------------------------------------------------------------------


class TestNephroRulesCrystals:
    def test_one_above_normal_score_correct(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Crystals at 6 (1 above normal max 5, weight 0.3) must produce score 0.3."""
        result = nephro_rules.evaluate({"crystals": 6})
        assert result.weighted_score == pytest.approx(0.3) # type: ignore[no-untyped-call]

    def test_far_above_normal_score_correct(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Crystals at 15 (10 above normal max 5, weight 0.3) must produce score 3.0."""
        result = nephro_rules.evaluate({"crystals": 15})
        assert result.weighted_score == pytest.approx(3.0) # type: ignore[no-untyped-call]

    def test_crystal_matched_particle_fields(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Matched particle for crystals must have correct fields."""
        result = nephro_rules.evaluate({"crystals": 10})
        assert len(result.matched_particles) == 1
        particle = result.matched_particles[0]
        assert particle.name == "crystals"
        assert particle.detected_count == 10
        assert particle.normal_range_max == 5
        assert particle.weight == pytest.approx(0.3) # type: ignore[no-untyped-call]
        assert particle.contribution == pytest.approx(1.5) # type: ignore[no-untyped-call]

    def test_contribution_formula_correct(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """contribution must equal (detected_count - normal_range_max) * weight."""
        result = nephro_rules.evaluate({"crystals": 8})
        particle = result.matched_particles[0]
        expected = (8 - 5) * 0.3
        assert particle.contribution == pytest.approx(expected) # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Erythrocytes above normal range
# ---------------------------------------------------------------------------


class TestNephroRulesErythrocytes:
    def test_erythrocytes_above_normal_contributes(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Erythrocytes at 5 (2 above normal max 3, weight 0.1) must produce score 0.2."""
        result = nephro_rules.evaluate({"erythrocytes": 5})
        assert result.weighted_score == pytest.approx(0.2) # type: ignore[no-untyped-call]

    def test_erythrocytes_contribution_formula(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """contribution must equal (detected_count - normal_range_max) * weight."""
        result = nephro_rules.evaluate({"erythrocytes": 8})
        particle = result.matched_particles[0]
        expected = (8 - 3) * 0.1
        assert particle.contribution == pytest.approx(expected) # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Multiple particles above normal
# ---------------------------------------------------------------------------


class TestNephroRulesMultipleParticles:
    def test_scores_accumulate_correctly(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Scores from multiple particles must accumulate correctly."""
        result = nephro_rules.evaluate({"crystals": 10, "erythrocytes": 5})
        # crystals: (10-5)*0.3 = 1.5, erythrocytes: (5-3)*0.1 = 0.2, total = 1.7
        assert result.weighted_score == pytest.approx(1.7) # type: ignore[no-untyped-call]

    def test_two_matched_particles_returned(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Both particles above normal range must appear in matched_particles."""
        result = nephro_rules.evaluate({"crystals": 10, "erythrocytes": 5})
        assert len(result.matched_particles) == 2


# ---------------------------------------------------------------------------
# Missing keys and unknown keys
# ---------------------------------------------------------------------------


class TestNephroRulesMissingKeys:
    def test_missing_particle_treated_as_zero(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """Missing particle key must be treated as count = 0."""
        result = nephro_rules.evaluate({})
        assert result.weighted_score == 0.0

    def test_unknown_key_ignored(self, nephro_rules: NephrolithiasisRules) -> None:
        """Unknown particle keys must be ignored."""
        result = nephro_rules.evaluate({"unknown_particle": 999})
        assert result.weighted_score == 0.0


# ---------------------------------------------------------------------------
# RuleResult fields
# ---------------------------------------------------------------------------


class TestNephroRuleResultFields:
    def test_condition_is_nephrolithiasis(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """RuleResult.condition must always be 'nephrolithiasis'."""
        result = nephro_rules.evaluate({})
        assert result.condition == "nephrolithiasis"

    def test_returns_rule_result_instance(
        self, nephro_rules: NephrolithiasisRules
    ) -> None:
        """evaluate() must return a RuleResult instance."""
        result = nephro_rules.evaluate({})
        assert isinstance(result, RuleResult)