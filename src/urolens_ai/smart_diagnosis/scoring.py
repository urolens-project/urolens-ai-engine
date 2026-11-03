"""
smart_diagnosis/scoring.py
--------------------------
Scoring logic for the Smart Diagnosis rule engine.

Maps a condition's weighted_score to a ProbabilityLevel (LOW / MODERATE / HIGH)
based on thresholds defined in config.yaml.

Public functions:
    compute_level(weighted_score, low_max, high_min) -> ProbabilityLevel
    cap_without_marker(level, matched_names, high_requires) -> ProbabilityLevel
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


def cap_without_marker(
    level: ProbabilityLevel,
    matched_names: set[str],
    high_requires: str | None,
) -> ProbabilityLevel:
    """
    Cap HIGH at MODERATE when the condition's primary marker is absent.

    Parameters
    ----------
    level : ProbabilityLevel
        Level from compute_level().
    matched_names : set[str]
        Names of particles that exceeded their normal range.
    high_requires : str | None
        Particle that must exceed its normal range for HIGH.
        Sourced from config.yaml high_requires. None disables the cap.

    Returns
    -------
    ProbabilityLevel
        MODERATE if level is HIGH and high_requires is not in matched_names,
        otherwise level unchanged.

    Notes
    -----
    - Stops a non-specific particle (erythrocytes) from reaching HIGH on
      its own through unbounded excess scoring.
    """
    if (
        high_requires is None
        or level != ProbabilityLevel.HIGH
        or high_requires in matched_names
    ):
        return level

    logger.debug(
        "cap_without_marker",
        extra={"high_requires": high_requires, "level": ProbabilityLevel.MODERATE.value},
    )
    return ProbabilityLevel.MODERATE
