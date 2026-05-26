"""
smart_diagnosis/scoring.py
--------------------------
Scoring logic for the Smart Diagnosis rule engine.

Maps a condition's weighted_score to a ProbabilityLevel (LOW / MODERATE / HIGH)
based on thresholds defined in config.yaml.

Public functions:
    compute_level(weighted_score, low_max, high_min) -> ProbabilityLevel
"""

from __future__ import annotations

from urolens_ai.schemas.smart_diagnosis import ProbabilityLevel
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)


def compute_level(
    weighted_score: float,
    low_max: float,
    high_min: float,
) -> ProbabilityLevel:
    """
    Map a weighted score to a probability level.

    Parameters
    ----------
    weighted_score : float
        The condition's weighted score from rule evaluation.
        Must be >= 0.0.
    low_max : float
        Score at or below this value maps to LOW.
        Sourced from config.yaml thresholds.low_max_score.
    high_min : float
        Score at or above this value maps to HIGH.
        Sourced from config.yaml thresholds.high_min_score.

    Returns
    -------
    ProbabilityLevel
        LOW    — weighted_score <= low_max
        HIGH   — weighted_score >= high_min
        MODERATE — weighted_score between low_max and high_min

    Notes
    -----
    - There is no cap on weighted_score — scores far above high_min
      still return HIGH.
    - Boundary values are inclusive:
        exactly low_max  -> LOW
        exactly high_min -> HIGH
    """
    if weighted_score <= low_max:
        level = ProbabilityLevel.LOW
    elif weighted_score >= high_min:
        level = ProbabilityLevel.HIGH
    else:
        level = ProbabilityLevel.MODERATE

    logger.debug(
        "compute_level",
        extra={
            "weighted_score": weighted_score,
            "low_max": low_max,
            "high_min": high_min,
            "level": level.value,
        },
    )

    return level