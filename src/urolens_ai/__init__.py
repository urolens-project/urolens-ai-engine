"""
urolens_ai/__init__.py
----------------------
Public API for the urolens-ai-engine package.

Only two symbols are exported from this package:

    infer(image_bytes)                 -> InferenceResult
    generate_smart_diagnosis(classification) -> SmartDiagnosisOutput

Everything else (preprocessing, YOLOv8 engine, rule files, scoring, evidence)
is an implementation detail and must NOT be imported directly by the Mobile
Developer's backend services.

Contract version: Sprint 0 — stubs only.
Full implementations land in EPIC-AI-03 (infer) and EPIC-AI-04 (generate_smart_diagnosis).
"""

from __future__ import annotations

import time

from urolens_ai.schemas.inference import InferenceResult
from urolens_ai.schemas.smart_diagnosis import SmartDiagnosisOutput
from urolens_ai.utils.exceptions import InferenceError, RuleEngineError
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["infer", "generate_smart_diagnosis"]


def infer(image_bytes: bytes) -> InferenceResult:
    """
    Run YOLOv8 inference on a urine sediment microscopy image.

    Accepts raw image bytes (JPEG or PNG), validates the image, runs
    YOLOv8 object detection, and returns a typed result containing particle
    counts, confidence scores, model version, and timing metadata.

    This is the sole entry point for all AI image analysis in UroLens.
    It is called exclusively from ``ai_integration_service.py`` in the
    ``urolens-backend`` project owned by the Mobile Developer.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG or PNG microscopy image.
        Minimum resolution: 640 × 480 pixels.
        Images below the minimum resolution are rejected before inference.

    Returns
    -------
    InferenceResult
        A Pydantic model containing:

        - ``model_version`` (str): The MODEL_VERSION environment variable
          value at inference time. Recorded in ``analysis_results.ai_findings``
          for auditability.
        - ``particles`` (dict[str, int]): Detected particle names mapped to
          their counts after confidence filtering and NMS.
          Keys match class names in ``models/yolov8/labels.txt``.
          Example: ``{"urinary_casts": 3, "erythrocytes": 7}``.
        - ``confidence_scores`` (dict[str, float]): Mean detection confidence
          per particle class, for particles that had at least one detection
          above the confidence threshold.
        - ``raw_detection_count`` (int): Total bounding boxes before confidence
          filtering. Useful for debugging threshold calibration.
        - ``filtered_detection_count`` (int): Bounding boxes remaining after
          confidence filtering. This is what ``particles`` counts are based on.
        - ``inference_time_ms`` (float): Wall-clock time from start of
          preprocessing to end of postprocessing, in milliseconds.
        - ``image_width`` (int): Width of the input image in pixels.
        - ``image_height`` (int): Height of the input image in pixels.

    Raises
    ------
    ImageValidationError
        Raised before any inference if the image fails validation.
        Possible ``code`` values:

        - ``"FORMAT_UNSUPPORTED"`` — image is not JPEG or PNG.
        - ``"RESOLUTION_TOO_LOW"`` — image is smaller than 640 × 480.
        - ``"CORRUPT_IMAGE"`` — image bytes cannot be decoded.

    InferenceError
        Raised if the YOLOv8 model fails during inference.
        Possible ``code`` values:

        - ``"MODEL_NOT_LOADED"`` — weights file missing or unreadable.
        - ``"INFERENCE_FAILED"`` — runtime error during model forward pass.

    Notes
    -----
    - EXIF metadata (GPS, device serial numbers) is stripped before inference.
      This is a patient-data privacy requirement.
    - The image is NOT resized before passing to YOLOv8. Ultralytics handles
      internal resizing. Pre-resizing changes particle sizes and degrades confidence.
    - NMS (Non-Maximum Suppression) is applied automatically by Ultralytics
      using the INFERENCE_IOU_THRESHOLD environment variable.
    - CPU inference is the default for MVP single-laboratory throughput.

    Examples
    --------
    Typical usage from ``ai_integration_service.py``::

        from urolens_ai import infer
        from urolens_ai.utils.exceptions import ImageValidationError, InferenceError

        try:
            result = infer(image_bytes)
            # result.particles  -> {"urinary_casts": 2, "erythrocytes": 5}
            # result.model_version -> "mvp-v1.0"
            # result.inference_time_ms -> 312.4
        except ImageValidationError as e:
            # e.code: "FORMAT_UNSUPPORTED" | "RESOLUTION_TOO_LOW" | "CORRUPT_IMAGE"
            # e.message: human-readable description with dimensions/format
            ...
        except InferenceError as e:
            # e.code: "MODEL_NOT_LOADED" | "INFERENCE_FAILED"
            ...
    """
    logger.debug("infer_stub_called", extra={"image_size_bytes": len(image_bytes)})
    # -----------------------------------------------------------------------
    # STUB — Sprint 0 only.
    # Full implementation: EPIC-AI-03, STORY-AI-04 (Sprint 4–5) and
    # STORY-AI-05 (Sprint 6).
    #
    # Implementation will follow this pipeline:
    #   1. preprocessing.validate_image(image_bytes)  — format, resolution, EXIF
    #   2. preprocessing.normalise(image_bytes)        — RGB numpy float32 array
    #   3. yolo_engine.get_engine().run(image_array)   — list[Detection]
    #   4. postprocessing.map_detections(detections)   — particle counts + scores
    #   5. return InferenceResult(...)
    # -----------------------------------------------------------------------
    raise NotImplementedError(
        "infer() is a Sprint 0 stub. "
        "Full implementation lands in STORY-AI-04 and STORY-AI-05 (Sprints 4–6)."
    )


def generate_smart_diagnosis(classification: dict[str, int]) -> SmartDiagnosisOutput:
    """
    Apply the Smart Diagnosis rule engine to a confirmed particle classification.

    Accepts the MedTech-confirmed particle counts (which may differ from the raw
    AI output if the MedTech applied manual overrides) and returns probability
    indicators (LOW / MODERATE / HIGH) for three target conditions, together with
    clinically transparent evidence attribution lists.

    This function is deterministic: identical ``classification`` input always
    produces identical output. This is required for clinical audit traceability.

    This is the sole entry point for rule-engine-based diagnosis support.
    It is called exclusively from ``smart_diagnosis_service.py`` in the
    ``urolens-backend`` project owned by the Mobile Developer, after the
    MedTech has confirmed the AI result.

    Parameters
    ----------
    classification : dict[str, int]
        Particle names mapped to confirmed counts.
        Keys must match class names in ``models/yolov8/labels.txt``.
        Unknown keys are ignored. Missing keys are treated as count = 0.
        Example::

            {
                "uric_acid_crystals": 12,
                "erythrocytes": 2,
                "urinary_casts": 0,
            }

    Returns
    -------
    SmartDiagnosisOutput
        A Pydantic model containing:

        - ``gout`` (ConditionScore): Level and evidence for Gout.
          Driven by uric acid crystals and ammonium biurate counts.
        - ``glomerulonephritis`` (ConditionScore): Level and evidence for
          Glomerulonephritis. Driven by RBC casts, dysmorphic RBCs,
          granular casts, and cellular casts.
        - ``nephrolithiasis`` (ConditionScore): Level and evidence for
          Nephrolithiasis (kidney stones). Driven by calcium oxalate,
          calcium phosphate, uric acid crystals, and struvite.
        - ``no_significant_indicators`` (bool): ``True`` when all three
          condition levels are LOW and all evidence lists are empty.
          This is a valid, successful output — not a failure state.
        - ``engine_version`` (str): The MODEL_VERSION environment variable
          value at call time, for auditability.

        Each ``ConditionScore`` contains:

        - ``level`` (ProbabilityLevel): LOW | MODERATE | HIGH.
        - ``weighted_score`` (float): Raw score before threshold mapping.
        - ``evidence`` (list[EvidenceItem]): Particles that drove the score,
          sorted by contribution descending. The highest contributor is
          ``contribution_role="primary"``; others are ``"supporting"``.

    Raises
    ------
    RuleEngineError
        Raised if the rule engine fails for any reason.
        Possible ``code`` values:

        - ``"INVALID_CLASSIFICATION"`` — classification input is not a dict
          or contains non-integer values.
        - ``"CONFIG_ERROR"`` — config.yaml is missing, malformed, or a
          required condition/particle entry is absent.
        - ``"RULE_EVALUATION_FAILED"`` — unexpected error during rule
          application. The Mobile Developer's ``smart_diagnosis_service.py``
          catches this and flags the result for Supervisor review without
          blocking the MedTech's confirmation action.

    Notes
    -----
    - The rule thresholds and particle weights are loaded from
      ``smart_diagnosis/config.yaml``, which is the single source of truth
      for all rule parameters. Thresholds must not be hardcoded in rule files.
    - config.yaml must be reviewed and signed off by the domain expert before
      Sprint 8. No rule implementation may be merged without that sign-off.
    - The rule engine is NOT a machine learning classifier. Every score is
      fully explainable: ``evidence`` lists the exact particles and their
      arithmetic contributions. This is a deliberate design decision required
      by the SRS.
    - The "no significant indicators" case (all LOW, empty evidence) must be
      handled explicitly. It is not an edge case — it is the expected output
      for a normal urinalysis result.

    Examples
    --------
    Typical usage from ``smart_diagnosis_service.py``::

        from urolens_ai import generate_smart_diagnosis
        from urolens_ai.utils.exceptions import RuleEngineError

        classification = {"uric_acid_crystals": 12, "erythrocytes": 1}

        try:
            output = generate_smart_diagnosis(classification)
            # output.gout.level          -> ProbabilityLevel.HIGH
            # output.gout.evidence[0].particle_name -> "uric_acid_crystals"
            # output.gout.evidence[0].contribution_role -> "primary"
            # output.no_significant_indicators -> False
        except RuleEngineError as e:
            # e.code: "INVALID_CLASSIFICATION" | "CONFIG_ERROR" | "RULE_EVALUATION_FAILED"
            # e.message: human-readable description
            ...

    Normal urinalysis (all within range)::

        output = generate_smart_diagnosis({"erythrocytes": 1, "leukocytes": 2})
        # output.no_significant_indicators -> True
        # output.gout.level               -> ProbabilityLevel.LOW
        # output.gout.evidence            -> []
    """
    logger.debug(
        "generate_smart_diagnosis_stub_called",
        extra={"particle_count": len(classification)},
    )
    # -----------------------------------------------------------------------
    # STUB — Sprint 0 only.
    # Full implementation: EPIC-AI-04, STORY-AI-07 (Sprint 7–8),
    # STORY-AI-08 (Sprint 8), STORY-AI-09 (Sprint 8).
    #
    # Implementation will follow this pipeline:
    #   1. Validate classification input type
    #   2. rule_engine.apply(gout_rules, classification)
    #   3. rule_engine.apply(gn_rules, classification)
    #   4. rule_engine.apply(nephro_rules, classification)
    #   5. scoring.compute_level(weighted_score) for each condition
    #   6. evidence.build_attribution(matched_particles) for each condition
    #   7. Set no_significant_indicators if all levels are LOW
    #   8. Return SmartDiagnosisOutput(...)
    # -----------------------------------------------------------------------
    raise NotImplementedError(
        "generate_smart_diagnosis() is a Sprint 0 stub. "
        "Full implementation lands in STORY-AI-07, STORY-AI-08, and STORY-AI-09 "
        "(Sprints 7–8)."
    )