"""
smart_diagnosis/rule_engine.py
--------------------------------
Rule engine orchestrator for the Smart Diagnosis pipeline.

Loads config.yaml, initialises all three condition rule evaluators,
applies them to the particle classification, and produces a complete
SmartDiagnosisOutput with probability levels and evidence attribution.

Public functions:
    run_rule_engine(classification) -> SmartDiagnosisOutput
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from urolens_ai.schemas.smart_diagnosis import (
    ConditionScore,
    ProbabilityLevel,
    SmartDiagnosisOutput,
)
from urolens_ai.smart_diagnosis.evidence import build_attribution
from urolens_ai.smart_diagnosis.rules.glomerulonephritis import GlomerulonephritisRules
from urolens_ai.smart_diagnosis.rules.gout import GoutRules
from urolens_ai.smart_diagnosis.rules.nephrolithiasis import NephrolithiasisRules
from urolens_ai.smart_diagnosis.scoring import compute_level
from urolens_ai.utils.exceptions import RuleEngineError
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)

_CONFIG_PATH: str = os.environ.get(
    "RULE_ENGINE_CONFIG_PATH",
    "src/urolens_ai/smart_diagnosis/config.yaml",
)


def _load_config() -> dict[str, Any]:
    """
    Load and parse config.yaml.

    Raises
    ------
    RuleEngineError
        code="CONFIG_ERROR" if the file is missing or malformed.
    """
    path = Path(_CONFIG_PATH)
    if not path.exists():
        raise RuleEngineError(
            code="CONFIG_ERROR",
            message=f"Rule engine config not found at '{_CONFIG_PATH}'.",
        )
    try:
        with open(path, "r") as f:
            config: dict[str, Any] = yaml.safe_load(f)
        return config
    except yaml.YAMLError as exc:
        raise RuleEngineError(
            code="CONFIG_ERROR",
            message=f"Failed to parse config.yaml: {exc}",
        ) from exc


def run_rule_engine(
    classification: dict[str, int],
) -> SmartDiagnosisOutput:
    """
    Apply all three condition rules to a particle classification.

    Parameters
    ----------
    classification : dict[str, int]
        Particle names mapped to confirmed counts.
        Missing keys are treated as 0. Unknown keys are ignored.

    Returns
    -------
    SmartDiagnosisOutput
        Probability levels and evidence for all three conditions.
        no_significant_indicators=True when all levels are LOW
        and all evidence lists are empty.

    Raises
    ------
    RuleEngineError
        code="CONFIG_ERROR" — config.yaml missing or malformed.
        code="RULE_EVALUATION_FAILED" — unexpected error during evaluation.
    """
    try:
        config = _load_config()
        conditions: dict[str, Any] = config["conditions"]
        display_names: dict[str, str] = config.get("display_names", {})

        # Initialise rule evaluators
        gout_rules = GoutRules(config=conditions["gout"])
        gn_rules = GlomerulonephritisRules(config=conditions["glomerulonephritis"])
        nephro_rules = NephrolithiasisRules(config=conditions["nephrolithiasis"])

        # Evaluate all three conditions
        gout_result = gout_rules.evaluate(classification)
        gn_result = gn_rules.evaluate(classification)
        nephro_result = nephro_rules.evaluate(classification)

        # Compute probability levels
        gout_level = compute_level(
            gout_result.weighted_score,
            conditions["gout"]["thresholds"]["low_max_score"],
            conditions["gout"]["thresholds"]["high_min_score"],
        )
        gn_level = compute_level(
            gn_result.weighted_score,
            conditions["glomerulonephritis"]["thresholds"]["low_max_score"],
            conditions["glomerulonephritis"]["thresholds"]["high_min_score"],
        )
        nephro_level = compute_level(
            nephro_result.weighted_score,
            conditions["nephrolithiasis"]["thresholds"]["low_max_score"],
            conditions["nephrolithiasis"]["thresholds"]["high_min_score"],
        )

        # Build evidence attribution
        gout_evidence = build_attribution(
            gout_result.matched_particles,
            conditions["gout"]["evidence_min_count"],
            display_names,
        )
        gn_evidence = build_attribution(
            gn_result.matched_particles,
            conditions["glomerulonephritis"]["evidence_min_count"],
            display_names,
        )
        nephro_evidence = build_attribution(
            nephro_result.matched_particles,
            conditions["nephrolithiasis"]["evidence_min_count"],
            display_names,
        )

        # Determine no_significant_indicators
        no_significant_indicators = (
            gout_level == ProbabilityLevel.LOW
            and gn_level == ProbabilityLevel.LOW
            and nephro_level == ProbabilityLevel.LOW
            and len(gout_evidence) == 0
            and len(gn_evidence) == 0
            and len(nephro_evidence) == 0
        )

        logger.info(
            "smart_diagnosis_completed",
            extra={
                "gout": gout_level.value,
                "glomerulonephritis": gn_level.value,
                "nephrolithiasis": nephro_level.value,
                "no_significant_indicators": no_significant_indicators,
            },
        )

        return SmartDiagnosisOutput(
            gout=ConditionScore(
                condition="gout",
                level=gout_level,
                weighted_score=gout_result.weighted_score,
                evidence=gout_evidence,
            ),
            glomerulonephritis=ConditionScore(
                condition="glomerulonephritis",
                level=gn_level,
                weighted_score=gn_result.weighted_score,
                evidence=gn_evidence,
            ),
            nephrolithiasis=ConditionScore(
                condition="nephrolithiasis",
                level=nephro_level,
                weighted_score=nephro_result.weighted_score,
                evidence=nephro_evidence,
            ),
            no_significant_indicators=no_significant_indicators,
            engine_version=os.environ.get("MODEL_VERSION", "unknown"),
        )

    except RuleEngineError:
        raise
    except Exception as exc:
        logger.error(
            "smart_diagnosis_failed",
            extra={"error_code": "RULE_EVALUATION_FAILED", "detail": str(exc)},
        )
        raise RuleEngineError(
            code="RULE_EVALUATION_FAILED",
            message=f"Unexpected error during rule evaluation: {exc}",
        ) from exc