"""
schemas/smart_diagnosis.py
--------------------------
Pydantic v2 output models for the UroLens Smart Diagnosis rule engine.

These schemas are part of the stable public contract defined in Sprint 0.
Any change to field names, types, or enum values requires a version bump
and written approval from the Mobile Developer before code is written.

Public contract surface:
    SmartDiagnosisOutput  — top-level return type of generate_smart_diagnosis()
    ConditionScore        — per-condition probability level + evidence
    EvidenceItem          — single particle's contribution to a condition score
    ProbabilityLevel      — LOW / MODERATE / HIGH enum
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProbabilityLevel(str, Enum):
    """
    Probability indicator for a target condition.

    Values
    ------
    LOW
        Particle counts are within or near normal ranges. No clinical flag
        is warranted based on the current rule thresholds.
    MODERATE
        Particle counts exceed normal ranges to a degree that warrants
        clinician attention. weighted_score is between low_max and high_min.
    HIGH
        Particle counts significantly exceed normal ranges.
        weighted_score is at or above high_min_score in config.yaml.
    """

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """
    A single particle class that contributed to a condition's probability score.

    Evidence items are sorted by contribution descending.  The particle with
    the highest contribution receives contribution_role="primary"; all others
    receive "supporting".

    Only particles whose detected_count >= evidence_min_count (from config.yaml)
    appear in the evidence list.
    """

    particle_name: str = Field(
        ...,
        description=(
            "Internal particle identifier matching the key used in the "
            "classification dict and labels.txt. "
            "Example: 'uric_acid_crystals'."
        ),
    )
    particle_display_name: str = Field(
        ...,
        description=(
            "Human-readable label for display in the Supervisor and physician "
            "portal screens. Example: 'Uric Acid Crystals'."
        ),
    )
    detected_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of this particle type confirmed in the MedTech's "
            "classification (post-override). This is the count that drove "
            "the rule evaluation, not the raw AI detection count."
        ),
    )
    normal_range_max: int = Field(
        ...,
        ge=0,
        description=(
            "Upper bound of the normal count range for this particle, "
            "as defined in config.yaml. Counts above this threshold "
            "contribute to the weighted score."
        ),
    )
    contribution_weight: float = Field(
        ...,
        gt=0.0,
        description=(
            "Per-particle weight from config.yaml that scales the excess "
            "count into the condition's weighted_score. "
            "Example: weight=1.0 means each excess particle adds 1.0 to the score."
        ),
    )
    contribution_score: float = Field(
        ...,
        ge=0.0,
        description=(
            "Actual score contribution of this particle: "
            "(detected_count - normal_range_max) * contribution_weight. "
            "Included for full auditability — clinicians can verify the "
            "arithmetic directly."
        ),
    )
    contribution_role: str = Field(
        ...,
        description=(
            "Role of this particle in the evidence list. "
            "'primary'   — highest contribution_score in the list. "
            "'supporting' — all other contributing particles. "
            "Exactly one item per evidence list will have role='primary'."
        ),
        pattern=r"^(primary|supporting)$",
    )


# ---------------------------------------------------------------------------
# Per-condition score
# ---------------------------------------------------------------------------


class ConditionScore(BaseModel):
    """
    Probability score and supporting evidence for a single target condition.

    Conditions evaluated by the rule engine:
        - gout
        - glomerulonephritis
        - nephrolithiasis

    When all particle counts are within normal ranges, level=LOW, weighted_score=0.0,
    and evidence is an empty list.
    """

    condition: str = Field(
        ...,
        description=(
            "Internal condition identifier. One of: "
            "'gout', 'glomerulonephritis', 'nephrolithiasis'."
        ),
    )
    level: ProbabilityLevel = Field(
        ...,
        description=(
            "Probability indicator derived from weighted_score against the "
            "thresholds defined in config.yaml for this condition."
        ),
    )
    weighted_score: float = Field(
        ...,
        ge=0.0,
        description=(
            "Sum of (excess_count * weight) across all particles that exceeded "
            "their normal_range_max. A score of 0.0 means no particle exceeded "
            "its normal range. Used by compute_level() to determine ProbabilityLevel."
        ),
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list[EvidenceItem],
        description=(
            "Ordered list of particles that contributed to this condition's score, "
            "sorted by contribution_score descending. Empty when weighted_score == 0.0 "
            "or when all contributing particles are below evidence_min_count."
        ),
    )
                                                                                    

# ---------------------------------------------------------------------------
# Top-level output
# ---------------------------------------------------------------------------


class SmartDiagnosisOutput(BaseModel):
    """
    Complete output of generate_smart_diagnosis(classification).

    This is the object persisted to the smart_diagnosis_outputs table by the
    Mobile Developer's smart_diagnosis_service.py.  All three condition scores
    and the evidence_map are always present, even when no indicators are raised.

    The Mobile Developer serialises this to JSONB.  Field names must not change
    without a version bump and written approval.
    """

    gout: ConditionScore = Field(
        ...,
        description="Probability score and evidence for Gout.",
    )
    glomerulonephritis: ConditionScore = Field(
        ...,
        description="Probability score and evidence for Glomerulonephritis.",
    )
    nephrolithiasis: ConditionScore = Field(
        ...,
        description="Probability score and evidence for Nephrolithiasis (kidney stones).",
    )
    no_significant_indicators: bool = Field(
        ...,
        description=(
            "True when all three condition scores are LOW and all evidence lists are "
            "empty — i.e. every particle count was within normal range. "
            "This is a valid clinical output (normal urinalysis), not a failure state. "
            "The backend logs this as a successful result."
        ),
    )
    engine_version: str = Field(
        ...,
        description=(
            "Identifier of the rule engine configuration that produced this output. "
            "Sourced from the MODEL_VERSION environment variable at runtime. "
            "Recorded alongside every result for audit traceability — if thresholds "
            "change between sprints, historical results remain traceable to the "
            "config version that produced them."
        ),
    )