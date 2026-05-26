"""
tests/integration/test_full_smart_diagnosis.py
-----------------------------------------------
Integration tests for generate_smart_diagnosis().

These tests exercise the full public API call — from classification dict
through rule_engine -> scoring -> evidence attribution -> SmartDiagnosisOutput.
They do NOT mock any internal components; every layer is live.

JSON fixtures in tests/fixtures/sample_classifications/ define the input
classifications and the _expected output for each scenario. The test suite
reads those fixtures and asserts against them, so adding a new fixture file
automatically adds a new parameterised test scenario.

Run with:
    pytest tests/integration/test_full_smart_diagnosis.py -v -m integration

Environment:
    RULE_ENGINE_CONFIG_PATH — overrides the default config.yaml location.
    MODEL_VERSION           — controls engine_version in output (default "unknown").

Tolerance:
    Floating-point score comparisons use pytest.approx(rel=1e-6) to guard
    against float rounding drift across Python versions.
"""

from __future__ import annotations
from typing import Any

import json
from pathlib import Path

import pytest

from urolens_ai import generate_smart_diagnosis
from urolens_ai.schemas.smart_diagnosis import (
    ConditionScore,
    ProbabilityLevel,
    SmartDiagnosisOutput,
)
from urolens_ai.utils.exceptions import RuleEngineError

# ---------------------------------------------------------------------------
# Marks and paths
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_classifications"


# ---------------------------------------------------------------------------
# Fixture loader — parameterises from JSON files automatically
# ---------------------------------------------------------------------------


def _load_fixture(filename: str) -> tuple[dict[str, int], dict[str, Any]]:
    """Read a sample_classifications JSON fixture and return (classification, expected)."""
    path = FIXTURES_DIR / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["classification"], data["_expected"]


def _all_fixture_files() -> list[str]:
    return sorted(p.name for p in FIXTURES_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Helper: assert a single ConditionScore against expected dict
# ---------------------------------------------------------------------------


def _assert_condition_score(
    actual: ConditionScore,
    expected: dict[str, Any],
    condition_name: str,
) -> None:
    assert actual.level == ProbabilityLevel(expected["level"]), (
        f"{condition_name}: expected level {expected['level']}, got {actual.level.value}"
    )
    assert actual.weighted_score == pytest.approx( # type: ignore[union-attr]
        expected["weighted_score"], rel=1e-6
    ), (
        f"{condition_name}: expected weighted_score {expected['weighted_score']}, "
        f"got {actual.weighted_score}"
    )

    expected_evidence = expected.get("evidence", [])
    assert len(actual.evidence) == len(expected_evidence), (
        f"{condition_name}: expected {len(expected_evidence)} evidence items, "
        f"got {len(actual.evidence)}"
    )

    for i, (ev_actual, ev_expected) in enumerate(
        zip(actual.evidence, expected_evidence)
    ):
        assert ev_actual.particle_name == ev_expected["particle_name"], (
            f"{condition_name} evidence[{i}]: expected particle_name "
            f"'{ev_expected['particle_name']}', got '{ev_actual.particle_name}'"
        )
        assert ev_actual.detected_count == ev_expected["detected_count"], (
            f"{condition_name} evidence[{i}]: expected detected_count "
            f"{ev_expected['detected_count']}, got {ev_actual.detected_count}"
        )
        assert ev_actual.normal_range_max == ev_expected["normal_range_max"], (
            f"{condition_name} evidence[{i}]: expected normal_range_max "
            f"{ev_expected['normal_range_max']}, got {ev_actual.normal_range_max}"
        )
        assert ev_actual.contribution_weight == pytest.approx( # type: ignore[union-attr]
            ev_expected["contribution_weight"], rel=1e-6
        ), (
            f"{condition_name} evidence[{i}]: expected contribution_weight "
            f"{ev_expected['contribution_weight']}, got {ev_actual.contribution_weight}"
        )
        assert ev_actual.contribution_score == pytest.approx( # type: ignore[union-attr]
            ev_expected["contribution_score"], rel=1e-6
        ), (
            f"{condition_name} evidence[{i}]: expected contribution_score "
            f"{ev_expected['contribution_score']}, got {ev_actual.contribution_score}"
        )
        assert ev_actual.contribution_role == ev_expected["contribution_role"], (
            f"{condition_name} evidence[{i}]: expected role "
            f"'{ev_expected['contribution_role']}', got '{ev_actual.contribution_role}'"
        )


# ===========================================================================
# CLASS 1 — Return type and schema contract
# ===========================================================================


class TestOutputSchema:
    """generate_smart_diagnosis() must always return a fully populated SmartDiagnosisOutput."""

    def test_returns_smart_diagnosis_output_instance(self) -> None:
        result = generate_smart_diagnosis({"crystals": 0})
        assert isinstance(result, SmartDiagnosisOutput)

    def test_all_three_conditions_always_present(self) -> None:
        result = generate_smart_diagnosis({})
        assert isinstance(result.gout, ConditionScore)
        assert isinstance(result.glomerulonephritis, ConditionScore)
        assert isinstance(result.nephrolithiasis, ConditionScore)

    def test_condition_names_are_correct(self) -> None:
        result = generate_smart_diagnosis({})
        assert result.gout.condition == "gout"
        assert result.glomerulonephritis.condition == "glomerulonephritis"
        assert result.nephrolithiasis.condition == "nephrolithiasis"

    def test_weighted_score_is_non_negative(self) -> None:
        result = generate_smart_diagnosis({"crystals": 20, "erythrocytes": 10})
        assert result.gout.weighted_score >= 0.0
        assert result.glomerulonephritis.weighted_score >= 0.0
        assert result.nephrolithiasis.weighted_score >= 0.0

    def test_level_is_valid_probability_level(self) -> None:
        result = generate_smart_diagnosis({"crystals": 20})
        valid_levels = {ProbabilityLevel.LOW, ProbabilityLevel.MODERATE, ProbabilityLevel.HIGH}
        assert result.gout.level in valid_levels
        assert result.glomerulonephritis.level in valid_levels
        assert result.nephrolithiasis.level in valid_levels

    def test_engine_version_is_string(self) -> None:
        result = generate_smart_diagnosis({})
        assert isinstance(result.engine_version, str)
        assert len(result.engine_version) > 0

    def test_engine_version_matches_model_version_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_VERSION", "rule-engine-v1.0-test")
        result = generate_smart_diagnosis({})
        assert result.engine_version == "rule-engine-v1.0-test"

    def test_no_significant_indicators_is_bool(self) -> None:
        result = generate_smart_diagnosis({})
        assert isinstance(result.no_significant_indicators, bool)


# ===========================================================================
# CLASS 2 — Normal / no-indicator scenarios
# ===========================================================================


class TestNoSignificantIndicators:
    """Counts within normal range must produce all-LOW output and no_significant_indicators=True."""

    def test_empty_classification_returns_all_low(self) -> None:
        result = generate_smart_diagnosis({})
        assert result.gout.level == ProbabilityLevel.LOW
        assert result.glomerulonephritis.level == ProbabilityLevel.LOW
        assert result.nephrolithiasis.level == ProbabilityLevel.LOW

    def test_empty_classification_no_significant_indicators_true(self) -> None:
        result = generate_smart_diagnosis({})
        assert result.no_significant_indicators is True

    def test_empty_classification_all_scores_zero(self) -> None:
        result = generate_smart_diagnosis({})
        assert result.gout.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]
        assert result.glomerulonephritis.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]
        assert result.nephrolithiasis.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]

    def test_empty_classification_all_evidence_empty(self) -> None:
        result = generate_smart_diagnosis({})
        assert result.gout.evidence == []
        assert result.glomerulonephritis.evidence == []
        assert result.nephrolithiasis.evidence == []

    def test_all_counts_at_normal_range_max_returns_all_low(self) -> None:
        """
        Boundary: detected_count == normal_range_max must NOT contribute to score.
        Scoring formula uses strict greater-than: if count > normal_max.
        """
        classification = {
            "crystals": 5,       # gout normal_range_max=5, nephro normal_range_max=5
            "erythrocytes": 3,   # gn normal_range_max=3, nephro normal_range_max=3
            "urinary_casts": 0,  # gn normal_range_max=0
        }
        result = generate_smart_diagnosis(classification)
        assert result.gout.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]
        assert result.glomerulonephritis.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]
        assert result.nephrolithiasis.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]
        assert result.no_significant_indicators is True

    def test_unknown_particle_keys_are_ignored(self) -> None:
        """
        Particles not referenced in config.yaml must be silently ignored.
        Elevated bacteria, leukocytes, yeast — none trigger any condition.
        """
        classification = {
            "bacteria": 50,
            "leukocytes": 30,
            "yeast": 10,
            "mucus_threads": 5,
            "sperm_cells": 2,
        }
        result = generate_smart_diagnosis(classification)
        assert result.no_significant_indicators is True
        assert result.gout.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]
        assert result.glomerulonephritis.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]
        assert result.nephrolithiasis.weighted_score == pytest.approx(0.0)  # type: ignore[union-attr]

    def test_no_significant_indicators_false_when_any_condition_not_low(self) -> None:
        """no_significant_indicators must be False if any condition is not LOW."""
        result = generate_smart_diagnosis({"crystals": 20})
        assert result.gout.level == ProbabilityLevel.HIGH
        assert result.no_significant_indicators is False


# ===========================================================================
# CLASS 3 — Gout scoring
# ===========================================================================


class TestGoutScoring:
    """Gout condition: crystals weight=1.0, normal_range_max=5, low_max=3.0, high_min=10.0."""

    def test_gout_low_boundary_score_equals_low_max(self) -> None:
        """
        Boundary: weighted_score == low_max_score (3.0) must return LOW.
        crystals=8 => (8-5)*1.0 = 3.0 exactly.
        compute_level uses <=, so exactly 3.0 is LOW.
        """
        result = generate_smart_diagnosis({"crystals": 8})
        assert result.gout.weighted_score == pytest.approx(3.0, rel=1e-6)  # type: ignore[union-attr]
        assert result.gout.level == ProbabilityLevel.LOW

    def test_gout_moderate_just_above_low_max(self) -> None:
        """
        crystals=9 => (9-5)*1.0 = 4.0 — above low_max(3.0), below high_min(10.0).
        """
        result = generate_smart_diagnosis({"crystals": 9})
        assert result.gout.weighted_score == pytest.approx(4.0, rel=1e-6)  # type: ignore[union-attr]
        assert result.gout.level == ProbabilityLevel.MODERATE

    def test_gout_high_boundary_score_equals_high_min(self) -> None:
        """
        Boundary: weighted_score == high_min_score (10.0) must return HIGH.
        crystals=15 => (15-5)*1.0 = 10.0.
        compute_level uses >=, so exactly 10.0 is HIGH.
        """
        result = generate_smart_diagnosis({"crystals": 15})
        assert result.gout.weighted_score == pytest.approx(10.0, rel=1e-6) # type: ignore[union-attr]
        assert result.gout.level == ProbabilityLevel.HIGH

    def test_gout_high_well_above_threshold(self) -> None:
        """crystals=20 => (20-5)*1.0 = 15.0 => HIGH."""
        result = generate_smart_diagnosis({"crystals": 20})
        assert result.gout.weighted_score == pytest.approx(15.0, rel=1e-6) # type: ignore[union-attr]
        assert result.gout.level == ProbabilityLevel.HIGH

    def test_gout_evidence_present_when_crystals_exceed_evidence_min_count(self) -> None:
        """
        evidence_min_count=2. crystals=8 >= 2, so evidence must contain crystals.
        """
        result = generate_smart_diagnosis({"crystals": 8})
        assert len(result.gout.evidence) == 1
        ev = result.gout.evidence[0]
        assert ev.particle_name == "crystals"
        assert ev.detected_count == 8
        assert ev.normal_range_max == 5
        assert ev.contribution_weight == pytest.approx(1.0, rel=1e-6) # type: ignore[union-attr]
        assert ev.contribution_score == pytest.approx(3.0, rel=1e-6) # type: ignore[union-attr]
        assert ev.contribution_role == "primary"

    def test_gout_evidence_empty_when_no_crystals(self) -> None:
        result = generate_smart_diagnosis({"crystals": 0})
        assert result.gout.evidence == []

    def test_gout_evidence_empty_when_crystals_below_evidence_min_count(self) -> None:
        """
        crystals=1 is within normal_range_max=5 so never contributes.
        Even if it did contribute (hypothetically), evidence_min_count=2
        would filter it out. This test ensures normal-range crystals
        produce no evidence.
        """
        result = generate_smart_diagnosis({"crystals": 1})
        assert result.gout.evidence == []

    def test_gout_does_not_score_non_crystal_particles(self) -> None:
        """Elevated erythrocytes and urinary_casts must not affect gout score."""
        result = generate_smart_diagnosis({
            "erythrocytes": 50,
            "urinary_casts": 20,
        })
        assert result.gout.weighted_score == pytest.approx(0.0) # type: ignore[union-attr]
        assert result.gout.level == ProbabilityLevel.LOW


# ===========================================================================
# CLASS 4 — Glomerulonephritis scoring
# ===========================================================================


class TestGlomerulonephritisScoring:
    """
    GN: urinary_casts weight=0.4 (max=0), erythrocytes weight=0.2 (max=3).
    low_max=2.0, high_min=10.0, evidence_min_count=1.
    """

    def test_gn_low_when_all_within_range(self) -> None:
        result = generate_smart_diagnosis({"urinary_casts": 0, "erythrocytes": 3})
        assert result.glomerulonephritis.level == ProbabilityLevel.LOW
        assert result.glomerulonephritis.weighted_score == pytest.approx(0.0) # type: ignore[union-attr]

    def test_gn_low_boundary_at_low_max(self) -> None:
        """
        urinary_casts=5 => 5*0.4=2.0 == low_max(2.0) => LOW (inclusive).
        """
        result = generate_smart_diagnosis({"urinary_casts": 5, "erythrocytes": 0})
        assert result.glomerulonephritis.weighted_score == pytest.approx(2.0, rel=1e-6) # type: ignore[union-attr]
        assert result.glomerulonephritis.level == ProbabilityLevel.LOW

    def test_gn_moderate_urinary_casts_and_erythrocytes(self) -> None:
        """
        urinary_casts=3, erythrocytes=8:
          casts: (3-0)*0.4=1.2, rbc: (8-3)*0.2=1.0 => total 2.2 => MODERATE.
        """
        result = generate_smart_diagnosis({"urinary_casts": 3, "erythrocytes": 8})
        assert result.glomerulonephritis.weighted_score == pytest.approx(2.2, rel=1e-6) # type: ignore[union-attr]
        assert result.glomerulonephritis.level == ProbabilityLevel.MODERATE

    def test_gn_high_boundary_at_high_min(self) -> None:
        """
        urinary_casts=25 => 25*0.4=10.0 == high_min(10.0) => HIGH (inclusive).
        """
        result = generate_smart_diagnosis({"urinary_casts": 25, "erythrocytes": 0})
        assert result.glomerulonephritis.weighted_score == pytest.approx(10.0, rel=1e-6) # type: ignore[union-attr]
        assert result.glomerulonephritis.level == ProbabilityLevel.HIGH

    def test_gn_high_casts_and_erythrocytes(self) -> None:
        """
        urinary_casts=25, erythrocytes=5:
          casts: 10.0, rbc: (5-3)*0.2=0.4 => total 10.4 => HIGH.
        """
        result = generate_smart_diagnosis({"urinary_casts": 25, "erythrocytes": 5})
        assert result.glomerulonephritis.weighted_score == pytest.approx(10.4, rel=1e-6) # type: ignore[union-attr]
        assert result.glomerulonephritis.level == ProbabilityLevel.HIGH

    def test_gn_evidence_primary_is_highest_contributor(self) -> None:
        """urinary_casts contributes more than erythrocytes — must be primary."""
        result = generate_smart_diagnosis({"urinary_casts": 3, "erythrocytes": 8})
        assert len(result.glomerulonephritis.evidence) == 2
        assert result.glomerulonephritis.evidence[0].particle_name == "urinary_casts"
        assert result.glomerulonephritis.evidence[0].contribution_role == "primary"
        assert result.glomerulonephritis.evidence[1].particle_name == "erythrocytes"
        assert result.glomerulonephritis.evidence[1].contribution_role == "supporting"

    def test_gn_evidence_respects_evidence_min_count(self) -> None:
        """
        evidence_min_count=1. urinary_casts=0 is at normal_range_max, so no contribution.
        erythrocytes=0 is below normal_range_max=3, so no contribution.
        Both within range => no evidence items.
        """
        result = generate_smart_diagnosis({"urinary_casts": 0, "erythrocytes": 0})
        assert result.glomerulonephritis.evidence == []

    def test_gn_does_not_score_crystals(self) -> None:
        """crystals are not a GN particle — must not affect GN score."""
        result = generate_smart_diagnosis({"crystals": 100})
        assert result.glomerulonephritis.weighted_score == pytest.approx(0.0) # type: ignore[union-attr]


# ===========================================================================
# CLASS 5 — Nephrolithiasis scoring
# ===========================================================================


class TestNephrolithiasisScoring:
    """
    Nephro: crystals weight=0.3 (max=5), erythrocytes weight=0.1 (max=3).
    low_max=3.0, high_min=15.0, evidence_min_count=2.
    """

    def test_nephro_low_when_all_within_range(self) -> None:
        result = generate_smart_diagnosis({"crystals": 5, "erythrocytes": 3})
        assert result.nephrolithiasis.level == ProbabilityLevel.LOW
        assert result.nephrolithiasis.weighted_score == pytest.approx(0.0) # type: ignore[union-attr]

    def test_nephro_moderate_crystals_elevated(self) -> None:
        """
        crystals=20, erythrocytes=10:
          crystals: (20-5)*0.3=4.5, rbc: (10-3)*0.1=0.7 => total 5.2.
          5.2 between low_max(3.0) and high_min(15.0) => MODERATE.
        """
        result = generate_smart_diagnosis({"crystals": 20, "erythrocytes": 10})
        assert result.nephrolithiasis.weighted_score == pytest.approx(5.2, rel=1e-6) # type: ignore[union-attr]
        assert result.nephrolithiasis.level == ProbabilityLevel.MODERATE

    def test_nephro_high_boundary_at_high_min(self) -> None:
        """
        high_min=15.0 is very hard to reach from crystals alone.
        crystals: 55 => (55-5)*0.3=15.0 exactly => HIGH.
        """
        result = generate_smart_diagnosis({"crystals": 55})
        assert result.nephrolithiasis.weighted_score == pytest.approx(15.0, rel=1e-6) # type: ignore[union-attr]
        assert result.nephrolithiasis.level == ProbabilityLevel.HIGH

    def test_nephro_evidence_sorted_by_contribution_descending(self) -> None:
        """
        crystals has higher contribution than erythrocytes.
        crystals should appear first (primary), erythrocytes second (supporting).
        """
        result = generate_smart_diagnosis({"crystals": 20, "erythrocytes": 10})
        assert len(result.nephrolithiasis.evidence) == 2
        assert result.nephrolithiasis.evidence[0].particle_name == "crystals"
        assert result.nephrolithiasis.evidence[0].contribution_role == "primary"
        assert result.nephrolithiasis.evidence[1].particle_name == "erythrocytes"
        assert result.nephrolithiasis.evidence[1].contribution_role == "supporting"

    def test_nephro_evidence_filtered_by_evidence_min_count(self) -> None:
        """
        evidence_min_count=2. erythrocytes=1 < evidence_min_count=2 — excluded.
        crystals=6 (> 5) with detected_count=6 >= 2 — included.
        """
        result = generate_smart_diagnosis({"crystals": 6, "erythrocytes": 1})
        particle_names = [ev.particle_name for ev in result.nephrolithiasis.evidence]
        assert "crystals" in particle_names
        assert "erythrocytes" not in particle_names


# ===========================================================================
# CLASS 6 — Evidence attribution invariants
# ===========================================================================


class TestEvidenceInvariants:
    """Cross-condition evidence correctness rules that always hold."""

    def test_exactly_one_primary_per_non_empty_evidence_list(self) -> None:
        """Each non-empty evidence list must have exactly one primary item."""
        result = generate_smart_diagnosis({"crystals": 20, "erythrocytes": 8, "urinary_casts": 4})
        for condition_score in [result.gout, result.glomerulonephritis, result.nephrolithiasis]:
            if condition_score.evidence:
                primaries = [
                    ev for ev in condition_score.evidence
                    if ev.contribution_role == "primary"
                ]
                assert len(primaries) == 1, (
                    f"{condition_score.condition}: expected exactly 1 primary, "
                    f"got {len(primaries)}"
                )

    def test_evidence_sorted_by_contribution_score_descending(self) -> None:
        """Evidence list must be strictly sorted by contribution_score descending."""
        result = generate_smart_diagnosis({"urinary_casts": 3, "erythrocytes": 8})
        ev = result.glomerulonephritis.evidence
        if len(ev) > 1:
            scores = [item.contribution_score for item in ev]
            assert scores == sorted(scores, reverse=True)

    def test_primary_is_first_item_in_evidence_list(self) -> None:
        """Primary contributor must always be the first item in the evidence list."""
        result = generate_smart_diagnosis({"urinary_casts": 5, "erythrocytes": 8})
        ev = result.glomerulonephritis.evidence
        if ev:
            assert ev[0].contribution_role == "primary"

    def test_contribution_score_arithmetic_is_correct(self) -> None:
        """
        contribution_score = (detected_count - normal_range_max) * contribution_weight.
        Verify the arithmetic explicitly for a known case.
        """
        # crystals=15, gout: weight=1.0, normal_max=5
        # expected contribution_score = (15-5)*1.0 = 10.0
        result = generate_smart_diagnosis({"crystals": 15})
        ev = result.gout.evidence[0]
        expected_contribution = (ev.detected_count - ev.normal_range_max) * ev.contribution_weight
        assert ev.contribution_score == pytest.approx(expected_contribution, rel=1e-9) # type: ignore[union-attr]

    def test_evidence_empty_when_weighted_score_zero(self) -> None:
        """If weighted_score == 0.0, evidence must be an empty list."""
        result = generate_smart_diagnosis({})
        for condition_score in [result.gout, result.glomerulonephritis, result.nephrolithiasis]:
            if condition_score.weighted_score == pytest.approx(0.0): # type: ignore[union-attr]
                assert condition_score.evidence == [], (
                    f"{condition_score.condition} has score 0.0 but non-empty evidence"
                )

    def test_no_significant_indicators_requires_all_empty_evidence(self) -> None:
        """
        no_significant_indicators=True implies all evidence lists are empty.
        """
        result = generate_smart_diagnosis({"crystals": 1, "erythrocytes": 1})
        if result.no_significant_indicators:
            assert result.gout.evidence == []
            assert result.glomerulonephritis.evidence == []
            assert result.nephrolithiasis.evidence == []


# ===========================================================================
# CLASS 7 — Input validation
# ===========================================================================


class TestInputValidation:
    """generate_smart_diagnosis() must validate its input and raise RuleEngineError."""

    def test_non_dict_input_raises_rule_engine_error(self) -> None:
        with pytest.raises(RuleEngineError) as exc_info:
            generate_smart_diagnosis(["crystals", 5])  # type: ignore[arg-type]
        assert exc_info.value.code == "INVALID_CLASSIFICATION"

    def test_none_input_raises_rule_engine_error(self) -> None:
        with pytest.raises(RuleEngineError) as exc_info:
            generate_smart_diagnosis(None)  # type: ignore[arg-type]
        assert exc_info.value.code == "INVALID_CLASSIFICATION"

    def test_negative_count_raises_rule_engine_error(self) -> None:
        with pytest.raises(RuleEngineError) as exc_info:
            generate_smart_diagnosis({"crystals": -1})
        assert exc_info.value.code == "INVALID_CLASSIFICATION"

    def test_float_value_raises_rule_engine_error(self) -> None:
        with pytest.raises(RuleEngineError) as exc_info:
            generate_smart_diagnosis({"crystals": 3.5})  # type: ignore[dict-item]
        assert exc_info.value.code == "INVALID_CLASSIFICATION"

    def test_string_value_raises_rule_engine_error(self) -> None:
        with pytest.raises(RuleEngineError) as exc_info:
            generate_smart_diagnosis({"crystals": "many"})  # type: ignore[dict-item]
        assert exc_info.value.code == "INVALID_CLASSIFICATION"

    def test_zero_count_is_valid(self) -> None:
        """Zero counts are valid — equivalent to a missing key."""
        result = generate_smart_diagnosis({"crystals": 0, "erythrocytes": 0})
        assert isinstance(result, SmartDiagnosisOutput)

    def test_missing_key_treated_as_zero(self) -> None:
        """
        Missing particle keys must behave identically to count=0.
        Both should produce identical output for particles not present.
        """
        result_missing = generate_smart_diagnosis({})
        result_explicit_zero = generate_smart_diagnosis({
            "crystals": 0,
            "erythrocytes": 0,
            "urinary_casts": 0,
        })
        assert result_missing.gout.weighted_score == pytest.approx( # type: ignore[union-attr]
            result_explicit_zero.gout.weighted_score
        ) 
        assert result_missing.glomerulonephritis.weighted_score == pytest.approx( # type: ignore[union-attr]
            result_explicit_zero.glomerulonephritis.weighted_score
        )
        assert result_missing.nephrolithiasis.weighted_score == pytest.approx(  # type: ignore[union-attr]
            result_explicit_zero.nephrolithiasis.weighted_score
        )


# ===========================================================================
# CLASS 8 — Fixture-driven parameterised tests
# ===========================================================================


class TestFixtureDriven:
    """
    Parameterised tests driven by JSON fixtures in tests/fixtures/sample_classifications/.

    Each fixture file defines:
        classification  — the input dict
        _expected       — the expected output (levels, scores, evidence)
        _description    — human-readable scenario description

    Adding a new .json file automatically adds a new test scenario.
    """

    @pytest.mark.parametrize("fixture_file", _all_fixture_files())
    def test_fixture_scenario(self, fixture_file: str) -> None:
        classification, expected = _load_fixture(fixture_file)
        result = generate_smart_diagnosis(classification)

        # Top-level assertions
        assert result.no_significant_indicators == expected["no_significant_indicators"], (
            f"[{fixture_file}] no_significant_indicators: "
            f"expected {expected['no_significant_indicators']}, "
            f"got {result.no_significant_indicators}"
        )

        # Per-condition assertions
        _assert_condition_score(result.gout, expected["gout"], f"[{fixture_file}] gout")
        _assert_condition_score(
            result.glomerulonephritis,
            expected["glomerulonephritis"],
            f"[{fixture_file}] glomerulonephritis",
        )
        _assert_condition_score(
            result.nephrolithiasis,
            expected["nephrolithiasis"],
            f"[{fixture_file}] nephrolithiasis",
        )