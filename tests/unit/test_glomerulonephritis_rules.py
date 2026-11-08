"""
tests/unit/test_glomerulonephritis_rules.py
--------------------------------------------
Unit tests for GlomerulonephritisRules.evaluate().

Coverage target: >= 90% branch coverage on rules/glomerulonephritis.py
"""

from __future__ import annotations
from typing import Any

import pytest

from urolens_ai.smart_diagnosis.rules.glomerulonephritis import GlomerulonephritisRules
from urolens_ai.smart_diagnosis.rules.gout import RuleResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GN_CONFIG: dict[str, Any] = {
    "particles": {
        "urinary_casts": {
            "weight": 0.4,
            "normal_range_max": 0,
            "display_name": "Urinary Casts",
        },
        "erythrocytes": {
            "weight": 0.2,
            "normal_range_max": 3,
            "display_name": "Erythrocytes (Red Blood Cells)",
        },
    },
    "thresholds": {
        "low_max_score": 2.0,
        "high_min_score": 10.0,
    },
    "evidence_min_count": 1,
}


@pytest.fixture
def gn_rules() -> GlomerulonephritisRules:
    return GlomerulonephritisRules(config=GN_CONFIG)


# ---------------------------------------------------------------------------
# All within normal range
# ---------------------------------------------------------------------------


class TestGNRulesWithinNormalRange:
    def test_all_zero_score_is_zero(self, gn_rules: GlomerulonephritisRules) -> None:
        """All particles at zero must produce weighted_score = 0.0."""
        result = gn_rules.evaluate({"urinary_casts": 0, "erythrocytes": 0})
        assert result.weighted_score == 0.0

    def test_erythrocytes_at_normal_max_score_zero(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """Erythrocytes at exactly normal_range_max (3) must not contribute."""
        result = gn_rules.evaluate({"urinary_casts": 0, "erythrocytes": 3})
        assert result.weighted_score == 0.0

    def test_empty_classification_score_zero(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """Empty classification must produce score = 0.0."""
        result = gn_rules.evaluate({})
        assert result.weighted_score == 0.0


# ---------------------------------------------------------------------------
# Urinary casts above normal range
# ---------------------------------------------------------------------------


class TestGNRulesCasts:
    def test_one_cast_contributes_to_score(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """1 urinary cast (above normal max 0, weight 0.4) must produce score 0.4."""
        result = gn_rules.evaluate({"urinary_casts": 1})
        assert result.weighted_score == pytest.approx(0.4) # type: ignore[no-untyped-call]

    def test_multiple_casts_score_correct(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """5 urinary casts (5 above normal max 0, weight 0.4) must produce score 2.0."""
        result = gn_rules.evaluate({"urinary_casts": 5})
        assert result.weighted_score == pytest.approx(2.0) # type: ignore[no-untyped-call]

    def test_cast_matched_particle_fields(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """Matched particle for urinary_casts must have correct fields."""
        result = gn_rules.evaluate({"urinary_casts": 3})
        assert len(result.matched_particles) == 1
        particle = result.matched_particles[0]
        assert particle.name == "urinary_casts"
        assert particle.detected_count == 3
        assert particle.normal_range_max == 0
        assert particle.weight == pytest.approx(0.4) # type: ignore[no-untyped-call]
        assert particle.contribution == pytest.approx(1.2) # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Erythrocytes above normal range
# ---------------------------------------------------------------------------


class TestGNRulesErythrocytes:
    def test_erythrocytes_above_normal_contributes(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """Erythrocytes at 5 (2 above normal max 3, weight 0.2) must produce score 0.4."""
        result = gn_rules.evaluate({"erythrocytes": 5})
        assert result.weighted_score == pytest.approx(0.4) # type: ignore[no-untyped-call]

    def test_erythrocytes_contribution_formula(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """contribution must equal (detected_count - normal_range_max) * weight."""
        result = gn_rules.evaluate({"erythrocytes": 8})
        particle = result.matched_particles[0]
        expected = (8 - 3) * 0.2
        assert particle.contribution == pytest.approx(expected) # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Multiple particles above normal
# ---------------------------------------------------------------------------


class TestGNRulesMultipleParticles:
    def test_scores_accumulate_correctly(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """Scores from multiple particles must accumulate correctly."""
        result = gn_rules.evaluate({"urinary_casts": 2, "erythrocytes": 5})
        # casts: (2-0)*0.4 = 0.8, erythrocytes: (5-3)*0.2 = 0.4, total = 1.2
        assert result.weighted_score == pytest.approx(1.2) # type: ignore[no-untyped-call]

    def test_two_matched_particles_returned(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """Both particles above normal range must appear in matched_particles."""
        result = gn_rules.evaluate({"urinary_casts": 2, "erythrocytes": 5})
        assert len(result.matched_particles) == 2


# ---------------------------------------------------------------------------
# Missing keys and unknown keys
# ---------------------------------------------------------------------------


class TestGNRulesMissingKeys:
    def test_missing_particle_treated_as_zero(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """Missing particle key must be treated as count = 0."""
        result = gn_rules.evaluate({})
        assert result.weighted_score == 0.0

    def test_unknown_key_ignored(self, gn_rules: GlomerulonephritisRules) -> None:
        """Unknown particle keys must be ignored."""
        result = gn_rules.evaluate({"unknown_particle": 999})
        assert result.weighted_score == 0.0


# ---------------------------------------------------------------------------
# RuleResult fields
# ---------------------------------------------------------------------------


class TestGNRuleResultFields:
    def test_condition_is_glomerulonephritis(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """RuleResult.condition must always be 'glomerulonephritis'."""
        result = gn_rules.evaluate({})
        assert result.condition == "glomerulonephritis"

    def test_returns_rule_result_instance(
        self, gn_rules: GlomerulonephritisRules
    ) -> None:
        """evaluate() must return a RuleResult instance."""
        result = gn_rules.evaluate({})
        assert isinstance(result, RuleResult)