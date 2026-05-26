"""
tests/unit/test_scoring.py
---------------------------
Unit tests for urolens_ai.smart_diagnosis.scoring.compute_level().

Tests every boundary condition as required by the guide.
Coverage target: >= 90% branch coverage on scoring.py
"""

from __future__ import annotations

from urolens_ai.schemas.smart_diagnosis import ProbabilityLevel
from urolens_ai.smart_diagnosis.scoring import compute_level


class TestComputeLevelBoundaries:
    def test_score_at_low_max_returns_low(self) -> None:
        """Score exactly at low_max must return LOW."""
        result = compute_level(3.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.LOW

    def test_score_below_low_max_returns_low(self) -> None:
        """Score below low_max must return LOW."""
        result = compute_level(1.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.LOW

    def test_score_zero_returns_low(self) -> None:
        """Score of 0.0 must return LOW."""
        result = compute_level(0.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.LOW

    def test_score_at_high_min_returns_high(self) -> None:
        """Score exactly at high_min must return HIGH."""
        result = compute_level(10.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.HIGH

    def test_score_above_high_min_returns_high(self) -> None:
        """Score above high_min must return HIGH."""
        result = compute_level(15.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.HIGH

    def test_score_far_above_high_min_returns_high(self) -> None:
        """Score far above high_min must still return HIGH — no cap."""
        result = compute_level(999.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.HIGH

    def test_score_between_thresholds_returns_moderate(self) -> None:
        """Score between low_max and high_min must return MODERATE."""
        result = compute_level(6.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.MODERATE

    def test_score_just_above_low_max_returns_moderate(self) -> None:
        """Score just above low_max must return MODERATE."""
        result = compute_level(3.1, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.MODERATE

    def test_score_just_below_high_min_returns_moderate(self) -> None:
        """Score just below high_min must return MODERATE."""
        result = compute_level(9.9, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.MODERATE


class TestComputeLevelReturnType:
    def test_returns_probability_level_instance(self) -> None:
        """compute_level() must return a ProbabilityLevel instance."""
        result = compute_level(0.0, low_max=3.0, high_min=10.0)
        assert isinstance(result, ProbabilityLevel)

    def test_gout_thresholds_low(self) -> None:
        """Gout thresholds (3.0/10.0) — score 0.0 must return LOW."""
        result = compute_level(0.0, low_max=3.0, high_min=10.0)
        assert result == ProbabilityLevel.LOW

    def test_gn_thresholds_any_cast_produces_above_low(self) -> None:
        """GN thresholds (2.0/10.0) — score 0.4 (1 cast) must return LOW still."""
        result = compute_level(0.4, low_max=2.0, high_min=10.0)
        assert result == ProbabilityLevel.LOW

    def test_nephro_thresholds_low(self) -> None:
        """Nephrolithiasis thresholds (3.0/15.0) — score 1.5 must return LOW."""
        result = compute_level(1.5, low_max=3.0, high_min=15.0)
        assert result == ProbabilityLevel.LOW