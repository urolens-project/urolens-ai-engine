"""
tests/unit/test_evidence.py
----------------------------
Unit tests for urolens_ai.smart_diagnosis.evidence.build_attribution().

Coverage target: >= 90% branch coverage on evidence.py
"""

from __future__ import annotations

import pytest

from urolens_ai.schemas.smart_diagnosis import EvidenceItem
from urolens_ai.smart_diagnosis.evidence import build_attribution
from urolens_ai.smart_diagnosis.rules.gout import MatchedParticle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "crystals": "Crystals",
    "urinary_casts": "Urinary Casts",
    "erythrocytes": "Erythrocytes (Red Blood Cells)",
}


def _make_particle(
    name: str,
    detected_count: int,
    contribution: float,
    normal_range_max: int = 5,
    weight: float = 1.0,
) -> MatchedParticle:
    return MatchedParticle(
        name=name,
        display_name=DISPLAY_NAMES.get(name, name),
        detected_count=detected_count,
        normal_range_max=normal_range_max,
        weight=weight,
        contribution=contribution,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestBuildAttributionEmpty:
    def test_no_matched_particles_returns_empty(self) -> None:
        """Empty matched_particles must return empty evidence list."""
        result = build_attribution([], evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert result == []

    def test_all_below_min_count_returns_empty(self) -> None:
        """Particles below evidence_min_count must be excluded."""
        particles = [_make_particle("crystals", detected_count=1, contribution=1.0)]
        result = build_attribution(particles, evidence_min_count=2, display_names=DISPLAY_NAMES)
        assert result == []


# ---------------------------------------------------------------------------
# Single particle
# ---------------------------------------------------------------------------


class TestBuildAttributionSingle:
    def test_single_particle_role_is_primary(self) -> None:
        """Single eligible particle must have contribution_role='primary'."""
        particles = [_make_particle("crystals", detected_count=10, contribution=5.0)]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert len(result) == 1
        assert result[0].contribution_role == "primary"

    def test_single_particle_fields_correct(self) -> None:
        """Single particle EvidenceItem must have correct field values."""
        particles = [_make_particle("crystals", detected_count=10, contribution=5.0)]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        item = result[0]
        assert item.particle_name == "crystals"
        assert item.particle_display_name == "Crystals"
        assert item.detected_count == 10
        assert item.contribution_score == pytest.approx(5.0)  # type: ignore[no-untyped-call]
        assert item.contribution_weight == pytest.approx(1.0)  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Multiple particles
# ---------------------------------------------------------------------------


class TestBuildAttributionMultiple:
    def test_multiple_particles_sorted_by_contribution(self) -> None:
        """Evidence list must be sorted by contribution_score descending."""
        particles = [
            _make_particle("erythrocytes", detected_count=8, contribution=2.0),
            _make_particle("crystals", detected_count=15, contribution=7.0),
        ]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert result[0].particle_name == "crystals"
        assert result[1].particle_name == "erythrocytes"

    def test_first_particle_is_primary(self) -> None:
        """First item in sorted list must have contribution_role='primary'."""
        particles = [
            _make_particle("erythrocytes", detected_count=8, contribution=2.0),
            _make_particle("crystals", detected_count=15, contribution=7.0),
        ]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert result[0].contribution_role == "primary"

    def test_remaining_particles_are_supporting(self) -> None:
        """All items after the first must have contribution_role='supporting'."""
        particles = [
            _make_particle("erythrocytes", detected_count=8, contribution=2.0),
            _make_particle("crystals", detected_count=15, contribution=7.0),
            _make_particle("urinary_casts", detected_count=3, contribution=1.2),
        ]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert result[1].contribution_role == "supporting"
        assert result[2].contribution_role == "supporting"

    def test_particle_below_min_count_excluded(self) -> None:
        """Particle below evidence_min_count must not appear in evidence list."""
        particles = [
            _make_particle("crystals", detected_count=10, contribution=5.0),
            _make_particle("erythrocytes", detected_count=1, contribution=0.5),
        ]
        result = build_attribution(particles, evidence_min_count=2, display_names=DISPLAY_NAMES)
        assert len(result) == 1
        assert result[0].particle_name == "crystals"


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------


class TestBuildAttributionDisplayNames:
    def test_display_name_from_config(self) -> None:
        """Display name must come from display_names mapping."""
        particles = [_make_particle("crystals", detected_count=10, contribution=5.0)]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert result[0].particle_display_name == "Crystals"

    def test_unknown_particle_uses_name_as_display(self) -> None:
        """Unknown particle name must fall back to internal name as display name."""
        particles = [_make_particle("unknown_particle", detected_count=10, contribution=5.0)]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert result[0].particle_display_name == "unknown_particle"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestBuildAttributionReturnType:
    def test_returns_list(self) -> None:
        """build_attribution() must return a list."""
        result = build_attribution([], evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert isinstance(result, list)

    def test_items_are_evidence_item_instances(self) -> None:
        """Each item must be an EvidenceItem instance."""
        particles = [_make_particle("crystals", detected_count=10, contribution=5.0)]
        result = build_attribution(particles, evidence_min_count=1, display_names=DISPLAY_NAMES)
        assert isinstance(result[0], EvidenceItem)