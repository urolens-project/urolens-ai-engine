"""
tests/unit/test_postprocessing.py
----------------------------------
Unit tests for urolens_ai.inference.postprocessing.

Tests detection aggregation, particle count accuracy, mean confidence
calculation, and edge cases. No model weights required.

Coverage target: ≥ 80% branch coverage on postprocessing.py
"""

from __future__ import annotations

import pytest

from urolens_ai.inference.postprocessing import map_detections
from urolens_ai.inference.yolo_engine import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    class_name: str,
    confidence: float,
    class_id: int = 0,
) -> Detection:
    """Create a Detection with minimal required fields."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(0.0, 0.0, 10.0, 10.0),
    )


# ---------------------------------------------------------------------------
# map_detections() — empty input
# ---------------------------------------------------------------------------


class TestMapDetectionsEmpty:
    def test_empty_list_returns_empty_dicts(self) -> None:
        """Empty detection list must return two empty dicts."""
        particles, confidence_scores = map_detections([])
        assert particles == {}
        assert confidence_scores == {}


# ---------------------------------------------------------------------------
# map_detections() — particle counts
# ---------------------------------------------------------------------------


class TestMapDetectionsCounts:
    def test_single_detection_count_is_one(self) -> None:
        """One detection of a class must produce count=1."""
        detections = [_make_detection("erythrocytes", 0.85)]
        particles, _ = map_detections(detections)
        assert particles["erythrocytes"] == 1

    def test_multiple_same_class_counts_correctly(self) -> None:
        """Three detections of the same class must produce count=3."""
        detections = [
            _make_detection("erythrocytes", 0.85),
            _make_detection("erythrocytes", 0.72),
            _make_detection("erythrocytes", 0.91),
        ]
        particles, _ = map_detections(detections)
        assert particles["erythrocytes"] == 3

    def test_multiple_classes_counted_independently(self) -> None:
        """Different classes must be counted independently."""
        detections = [
            _make_detection("erythrocytes", 0.85),
            _make_detection("erythrocytes", 0.72),
            _make_detection("leukocytes", 0.91),
            _make_detection("urinary-casts", 0.65),
            _make_detection("urinary-casts", 0.78),
        ]
        particles, _ = map_detections(detections)
        assert particles["erythrocytes"] == 2
        assert particles["leukocytes"] == 1
        assert particles["urinary-casts"] == 2

    def test_only_detected_classes_in_output(self) -> None:
        """Classes with zero detections must not appear in output."""
        detections = [_make_detection("erythrocytes", 0.85)]
        particles, _ = map_detections(detections)
        assert "leukocytes" not in particles
        assert "bacteria" not in particles


# ---------------------------------------------------------------------------
# map_detections() — confidence scores
# ---------------------------------------------------------------------------


class TestMapDetectionsConfidence:
    def test_single_detection_confidence_exact(self) -> None:
        """Single detection confidence must be returned as-is."""
        detections = [_make_detection("erythrocytes", 0.85)]
        _, confidence_scores = map_detections(detections)
        assert confidence_scores["erythrocytes"] == pytest.approx(0.85, abs=1e-4) # type: ignore

    def test_mean_confidence_computed_correctly(self) -> None:
        """Mean confidence must be the average of all detections for that class."""
        detections = [
            _make_detection("erythrocytes", 0.80),
            _make_detection("erythrocytes", 0.90),
        ]
        _, confidence_scores = map_detections(detections)
        assert confidence_scores["erythrocytes"] == pytest.approx(0.85, abs=1e-4) # type: ignore

    def test_confidence_rounded_to_4_decimal_places(self) -> None:
        """Confidence scores must be rounded to 4 decimal places."""
        detections = [
            _make_detection("erythrocytes", 0.8001),
            _make_detection("erythrocytes", 0.8002),
            _make_detection("erythrocytes", 0.8003),
        ]
        _, confidence_scores = map_detections(detections)
        # Result should be rounded to 4 decimal places
        assert len(str(confidence_scores["erythrocytes"]).split(".")[-1]) <= 4

    def test_confidence_keys_match_particle_keys(self) -> None:
        """confidence_scores keys must exactly match particles keys."""
        detections = [
            _make_detection("erythrocytes", 0.85),
            _make_detection("leukocytes", 0.91),
        ]
        particles, confidence_scores = map_detections(detections)
        assert set(particles.keys()) == set(confidence_scores.keys())


# ---------------------------------------------------------------------------
# map_detections() — return types
# ---------------------------------------------------------------------------


class TestMapDetectionsReturnTypes:
    def test_particles_values_are_integers(self) -> None:
        """Particle counts must be integers."""
        detections = [_make_detection("erythrocytes", 0.85)]
        particles, _ = map_detections(detections)
        assert isinstance(particles["erythrocytes"], int)

    def test_confidence_values_are_floats(self) -> None:
        """Confidence scores must be floats."""
        detections = [_make_detection("erythrocytes", 0.85)]
        _, confidence_scores = map_detections(detections)
        assert isinstance(confidence_scores["erythrocytes"], float)

    def test_returns_tuple_of_two_dicts(self) -> None:
        """map_detections() must return a tuple of (dict, dict)."""
        result = map_detections([])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)