"""
smart_diagnosis/rules/nephrolithiasis.py
-----------------------------------------
Nephrolithiasis condition rule evaluator.

Evaluates crystal and erythrocyte counts against clinically reviewed
thresholds from config.yaml.

Domain expert reviewed: Yes — Medical Technologist sign-off obtained.
Weights reduced and thresholds raised significantly per MedTech review —
Nephrolithiasis cannot be reliably diagnosed from urine sediment alone.
Imaging (ultrasound, CT) is required for definitive diagnosis.
This condition is retained as a supportive indicator only.
See docs/rule_engine_design.md Section 9.
"""

from __future__ import annotations

from typing import Any

from urolens_ai.smart_diagnosis.rules.gout import MatchedParticle, RuleResult
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)


class NephrolithiasisRules:
    """
    Rule evaluator for Nephrolithiasis condition.

    Evaluates crystal and erythrocyte counts against significantly reduced
    weights and raised thresholds per Medical Technologist review.
    This condition is a supportive indicator only — definitive diagnosis
    requires imaging confirmation.

    Parameters
    ----------
    config : dict
        The 'nephrolithiasis' section from config.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.particles: dict[str, Any] = config["particles"]
        self.thresholds: dict[str, Any] = config["thresholds"]
        self.evidence_min_count: int = config["evidence_min_count"]

    def evaluate(self, classification: dict[str, int]) -> RuleResult:
        """
        Evaluate particle counts against Nephrolithiasis rules.

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
        weighted_score = 0.0
        matched_particles: list[MatchedParticle] = []

        for particle_name, particle_config in self.particles.items():
            detected_count = classification.get(particle_name, 0)
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
            "nephrolithiasis_rules_evaluated",
            extra={
                "weighted_score": weighted_score,
                "matched_count": len(matched_particles),
            },
        )

        return RuleResult(
            condition="nephrolithiasis",
            weighted_score=weighted_score,
            matched_particles=matched_particles,
        )