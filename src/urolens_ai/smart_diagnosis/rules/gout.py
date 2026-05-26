"""
smart_diagnosis/rules/gout.py
------------------------------
Gout condition rule evaluator.

Evaluates particle counts against clinically reviewed thresholds from
config.yaml and returns a RuleResult containing the weighted score and
matched particles for scoring and evidence attribution.

Domain expert reviewed: Yes — Medical Technologist sign-off obtained.
See docs/rule_engine_design.md Section 9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared dataclasses — used across all rule files
# ---------------------------------------------------------------------------


@dataclass
class MatchedParticle:
    """
    A particle that exceeded its normal range and contributed to a condition score.

    Attributes
    ----------
    name : str
        Internal particle key matching config.yaml and labels.txt.
    display_name : str
        Human-readable label for evidence attribution UI.
    detected_count : int
        Confirmed particle count from the MedTech classification.
    normal_range_max : int
        Upper bound of normal range from config.yaml.
    weight : float
        Per-particle weight from config.yaml.
    contribution : float
        (detected_count - normal_range_max) * weight.
    """

    name: str
    display_name: str
    detected_count: int
    normal_range_max: int
    weight: float
    contribution: float


@dataclass
class RuleResult:
    """
    Output of a single condition rule evaluation.

    Attributes
    ----------
    condition : str
        Condition identifier — 'gout', 'glomerulonephritis', or 'nephrolithiasis'.
    weighted_score : float
        Sum of contributions across all matched particles.
    matched_particles : list[MatchedParticle]
        Particles that exceeded their normal range.
    """

    condition: str
    weighted_score: float
    matched_particles: list[MatchedParticle] = field(
        default_factory=lambda: []
    )


# ---------------------------------------------------------------------------
# GoutRules
# ---------------------------------------------------------------------------


class GoutRules:
    """
    Rule evaluator for Gout condition.

    Evaluates crystal counts against clinically reviewed thresholds.
    Elevated crystal counts indicate hyperuricemia and potential gout risk.

    Parameters
    ----------
    config : dict
        The 'gout' section from config.yaml, containing particles,
        thresholds, and evidence_min_count.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.particles: dict[str, Any] = config["particles"]
        self.thresholds: dict[str, Any] = config["thresholds"]
        self.evidence_min_count: int = config["evidence_min_count"]

    def evaluate(self, classification: dict[str, int]) -> RuleResult:
        """
        Evaluate particle counts against Gout rules.

        Parameters
        ----------
        classification : dict[str, int]
            Particle names mapped to confirmed counts.
            Missing keys are treated as count = 0.

        Returns
        -------
        RuleResult
            weighted_score = 0.0 and empty matched_particles when all
            counts are within normal range.
        """
        weighted_score: float = 0.0
        matched_particles: list[MatchedParticle] = []

        for particle_name, particle_config in self.particles.items():
            particle_name: str
            particle_config: dict[str, Any]
            detected_count: int = classification.get(particle_name, 0)
            normal_max: int = particle_config["normal_range_max"]
            weight: float = particle_config["weight"]
            display_name: str = particle_config["display_name"]

            if detected_count > normal_max:
                excess = detected_count - normal_max
                contribution = excess * weight
                weighted_score += contribution
                matched_particles.append(
                    MatchedParticle(
                        name=particle_name,
                        display_name=display_name,
                        detected_count=detected_count,
                        normal_range_max=normal_max,
                        weight=weight,
                        contribution=contribution,
                    )
                )

        logger.debug(
            "gout_rules_evaluated",
            extra={
                "weighted_score": weighted_score,
                "matched_count": len(matched_particles),
            },
        )

        return RuleResult(
            condition="gout",
            weighted_score=weighted_score,
            matched_particles=matched_particles,
        )