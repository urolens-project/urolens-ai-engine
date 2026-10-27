"""
inference/postprocessing.py
---------------------------
Maps raw YOLOv8 Detection objects to typed particle counts and confidence scores.

Called by infer() after yolo_engine.run() returns a list of Detection objects.
Aggregates detections by class name, filters by confidence threshold, and
produces the particles and confidence_scores dicts for InferenceResult.

Public functions:
    map_detections(detections)       — aggregates Detection list into particle counts
    to_output_detections(detections) — converts detections to the public box schema
"""

from __future__ import annotations

from collections import defaultdict

from urolens_ai.inference.yolo_engine import Detection
from urolens_ai.schemas.inference import Detection as DetectionOutput
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)


def particle_name(class_name: str) -> str:
    """
    Convert a model class name to the particle key used in every output.

    labels.txt uses dashes ("urinary-casts"); the particles dict, the boxes and
    the rule config all use underscores ("urinary_casts"). Converting here, and
    only here, keeps counts and boxes keyed identically.
    """
    return class_name.replace("-", "_")


def map_detections(
    detections: list[Detection],
) -> tuple[dict[str, int], dict[str, float]]:
    """
    Aggregate a list of Detection objects into particle counts and mean confidences.

    Parameters
    ----------
    detections : list[Detection]
        Raw detections from YOLOEngine.run(). May be empty.

    Returns
    -------
    particles : dict[str, int]
        Particle class names mapped to their detection counts.
        Only classes with at least one detection are included.
        Example: {"erythrocytes": 5, "urinary-casts": 2}

    confidence_scores : dict[str, float]
        Particle class names mapped to their mean confidence score.
        Only classes present in particles are included.
        Scores are rounded to 4 decimal places.
        Example: {"erythrocytes": 0.8341, "urinary-casts": 0.7102}

    Notes
    -----
    - NMS is applied upstream by Ultralytics before detections reach this function.
      Duplicate suppression is already handled.
    - If detections is empty, both returned dicts are empty.
    """
    if not detections:
        logger.debug("map_detections_empty", extra={"detection_count": 0})
        return {}, {}

    # Accumulate counts and confidence scores per class
    counts: dict[str, int] = defaultdict(int)
    confidences: dict[str, list[float]] = defaultdict(list)

    for detection in detections:
        name = particle_name(detection.class_name)
        counts[name] += 1
        confidences[name].append(detection.confidence)

    # Compute mean confidence per class
    particles: dict[str, int] = dict(counts)
    confidence_scores: dict[str, float] = {
        class_name: round(sum(scores) / len(scores), 4)
        for class_name, scores in confidences.items()
    }

    logger.debug(
        "map_detections_completed",
        extra={
            "total_detections": len(detections),
            "unique_classes": len(particles),
        },
    )

    return particles, confidence_scores

def to_output_detections(detections: list[Detection]) -> list[DetectionOutput]:
    """
    Convert engine detections to the public Detection schema.

    Parameters
    ----------
    detections : list[Detection]
        The same list passed to map_detections(), so every box here is
        counted in particles and vice versa.

    Returns
    -------
    list[DetectionOutput]
        One item per detection, in the engine's order, with class_name
        converted by particle_name() and bbox as [x1, y1, x2, y2] in pixels
        of the upright (EXIF-rotated) image.
    """
    return [
        DetectionOutput(
            class_id=detection.class_id,
            class_name=particle_name(detection.class_name),
            confidence=detection.confidence,
            bbox=list(detection.bbox),
        )
        for detection in detections
    ]
