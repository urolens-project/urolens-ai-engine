"""
urolens_ai/__init__.py
----------------------
Public API for the urolens-ai-engine package.

Only two symbols are exported from this package:

    infer(image_bytes)                       -> InferenceResult
    generate_smart_diagnosis(classification) -> SmartDiagnosisOutput

Everything else (preprocessing, YOLOv8 engine, rule files, scoring, evidence)
is an implementation detail and must NOT be imported directly by the Mobile
Developer's backend services.
"""

from __future__ import annotations

import math
import os
import time

from urolens_ai.inference.input_gate import check_input
from urolens_ai.inference.postprocessing import map_detections, to_output_detections
from urolens_ai.inference.preprocessing import normalise, validate_image
from urolens_ai.inference.yolo_engine import get_engine
from urolens_ai.schemas.inference import InferenceResult
from urolens_ai.schemas.smart_diagnosis import SmartDiagnosisOutput
from urolens_ai.smart_diagnosis.rule_engine import run_rule_engine
from urolens_ai.utils.exceptions import InferenceError, RuleEngineError
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["infer", "generate_smart_diagnosis"]


def infer(image_bytes: bytes) -> InferenceResult:
    """
    Run YOLOv8 inference on a urine sediment microscopy image.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG or PNG microscopy image.
        Minimum resolution: 640 × 480 pixels.

    Returns
    -------
    InferenceResult
        Particle counts, the counted boxes, confidence scores, model version,
        and timing metadata.

    Raises
    ------
    ImageValidationError
        code="FORMAT_UNSUPPORTED", "RESOLUTION_TOO_LOW", or "CORRUPT_IMAGE";
        "IMAGE_EXPOSURE" or "NOT_MICROSCOPY" if the input gate rejects it.
    InferenceError
        code="MODEL_NOT_LOADED" or "INFERENCE_FAILED".
    """
    logger.debug("infer_started", extra={"image_size_bytes": len(image_bytes)})
    start = time.perf_counter()

    # Step 1 — Validate image (format, resolution, corruption, EXIF strip)
    validate_image(image_bytes)

    # Step 2 — Normalise to float32 RGB numpy array. normalise() applies the
    # EXIF rotation, so the size is read from its output: the boxes are in
    # this upright image's pixels, and a sideways phone photo would otherwise
    # report width and height swapped.
    image_array = normalise(image_bytes)
    image_height, image_width = image_array.shape[:2]

    # Step 3 — Reject non-microscopy and unusable images before the detector,
    # which would otherwise label *something* on any photo
    check_input(image_array)

    # Step 4 — Run YOLOv8 inference
    engine = get_engine()

    try:
        detections = engine.run(image_array)
    except InferenceError:
        raise
    except Exception as exc:
        logger.error(
            "inference_failed",
            extra={"error_code": "INFERENCE_FAILED", "detail": str(exc)},
        )
        raise InferenceError(
            code="INFERENCE_FAILED",
            message=f"Unexpected error during inference: {exc}",
        ) from exc

    raw_detection_count = len(detections)

    # Step 5 — Map detections to particle counts and confidence scores
    particles, confidence_scores = map_detections(detections)
    output_detections = to_output_detections(detections)
    filtered_detection_count = sum(particles.values())

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "inference_completed",
        extra={
            "particle_count": len(particles),
            "duration_ms": round(elapsed_ms, 2),
        },
    )

    return InferenceResult(
        model_version=os.environ.get("MODEL_VERSION", "unknown"),
        particles=particles,
        confidence_scores=confidence_scores,
        detections=output_detections,
        raw_detection_count=raw_detection_count,
        filtered_detection_count=filtered_detection_count,
        inference_time_ms=round(elapsed_ms, 2),
        image_width=image_width,
        image_height=image_height,
    )


def generate_smart_diagnosis(classification: dict[str, float]) -> SmartDiagnosisOutput:
    """
    Apply the Smart Diagnosis rule engine to a confirmed particle classification.

    Parameters
    ----------
    classification : dict[str, float]
        Particle names mapped to confirmed counts. With multi-field capture
        these are averages per field (e.g. 2.4) and must be passed unrounded;
        whole-number counts from a single image are accepted as before.
        Unknown keys are ignored. Missing keys are treated as count = 0.

    Returns
    -------
    SmartDiagnosisOutput
        Probability scores and evidence for Gout, Glomerulonephritis,
        and Nephrolithiasis.

    Raises
    ------
    RuleEngineError
        code="INVALID_CLASSIFICATION", "CONFIG_ERROR", or "RULE_EVALUATION_FAILED".
    """

    # Validate input type — classification must be a dict
    if not isinstance(classification, dict): # type: ignore[arg-type]
        raise RuleEngineError(
            code="INVALID_CLASSIFICATION",
            message=(
                f"classification must be a dict[str, float], "
                f"got {type(classification).__name__}."
            ),
        )

    # Validate all values are finite, non-negative numbers. bool is a subclass
    # of int, so it is rejected explicitly rather than read as 0 or 1.
    for key, value in classification.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuleEngineError(
                code="INVALID_CLASSIFICATION",
                message=(
                    f"All classification values must be finite, non-negative numbers. "
                    f"Invalid value for '{key}': {value!r}."
                ),
            )

    start = time.perf_counter()

    result = run_rule_engine(classification)

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "generate_smart_diagnosis_completed",
        extra={
            "gout_level": result.gout.level.value,
            "glomerulonephritis_level": result.glomerulonephritis.level.value,
            "nephrolithiasis_level": result.nephrolithiasis.level.value,
            "no_significant_indicators": result.no_significant_indicators,
            "duration_ms": round(elapsed_ms, 2),
        },
    )

    return result