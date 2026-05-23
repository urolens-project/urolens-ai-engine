"""
inference/postprocessing.py
---------------------------
Maps raw YOLOv8 Detection objects to typed particle counts and confidence scores.

Called by infer() after yolo_engine.run() returns a list of Detection objects.
Aggregates detections by class name, filters by confidence threshold, and
produces the particles and confidence_scores dicts for InferenceResult.

Public functions:
    map_detections(detections)  — aggregates Detection list into particle counts
"""

from __future__ import annotations

from collections import defaultdict

from urolens_ai.inference.yolo_engine import Detection
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)


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
        counts[detection.class_name] += 1
        confidences[detection.class_name].append(detection.confidence)

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