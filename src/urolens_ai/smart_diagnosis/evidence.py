"""
smart_diagnosis/evidence.py
----------------------------
Evidence attribution for the Smart Diagnosis rule engine.

Generates a sorted, role-annotated list of EvidenceItem objects from the
matched particles produced by a rule evaluation. This provides full clinical
transparency — clinicians can see exactly which particles drove each
probability indicator and by how much.

Public functions:
    build_attribution(matched_particles, evidence_min_count, display_names) -> list[EvidenceItem]
"""

from __future__ import annotations

from urolens_ai.schemas.smart_diagnosis import EvidenceItem
from urolens_ai.smart_diagnosis.rules.gout import MatchedParticle
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)


def build_attribution(
    matched_particles: list[MatchedParticle],
    evidence_min_count: int,
    display_names: dict[str, str],
) -> list[EvidenceItem]:
    """
    Build a sorted, role-annotated evidence list from matched particles.

    Parameters
    ----------
    matched_particles : list[MatchedParticle]
        Particles that exceeded their normal range, from RuleResult.
    evidence_min_count : int
        Minimum detected_count for a particle to appear in the evidence list.
        Particles below this count are excluded even if they contributed
        to the weighted score.
    display_names : dict[str, str]
        Mapping from internal particle names to human-readable display names.
        Sourced from config.yaml display_names section.

    Returns
    -------
    list[EvidenceItem]
        Particles sorted by contribution_score descending.
        The first item has contribution_role="primary".
        All others have contribution_role="supporting".
        Empty list when matched_particles is empty or all particles
        are below evidence_min_count.

    Notes
    -----
    - Sorting is by contribution descending — the particle that drove
      the score the most appears first.
    - display_names fallback: if a particle name is not in display_names,
      the internal name is used as the display name.
    """
    if not matched_particles:
        return []

    # Filter particles below evidence_min_count
    eligible = [
        p for p in matched_particles
        if p.detected_count >= evidence_min_count
    ]

    if not eligible:
        return []

    # Sort by contribution descending
    sorted_particles = sorted(eligible, key=lambda p: p.contribution, reverse=True)

    items: list[EvidenceItem] = []
    for i, particle in enumerate(sorted_particles):
        role = "primary" if i == 0 else "supporting"
        display_name = display_names.get(particle.name, particle.name)

        items.append(
            EvidenceItem(
                particle_name=particle.name,
                particle_display_name=display_name,
                detected_count=particle.detected_count,
                normal_range_max=particle.normal_range_max,
                contribution_weight=particle.weight,
                contribution_score=particle.contribution,
                contribution_role=role,
            )
        )

    logger.debug(
        "build_attribution_completed",
        extra={
            "eligible_count": len(eligible),
            "evidence_count": len(items),
        },
    )

    return items