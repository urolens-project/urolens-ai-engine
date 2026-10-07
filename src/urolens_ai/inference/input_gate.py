"""
inference/input_gate.py
-----------------------
Reject images that are not usable urine microscopy before the detector runs.

The detector only knows the ten particle classes, so on any other photo it still
labels *something* -- a classroom selfie came back as "2 sperm cells" because
plain bright walls look like an empty microscope field. This gate runs two checks:

    1. Exposure   -- near-black or blown-out frames (lens cap on, light off,
                     flash). Thresholds are deliberately loose: real slides
                     range from mean 43 to 233, and OpenUrine eyepiece images
                     have up to ~40% black vignette.
    2. Content    -- a 2-class classifier (slide / not_slide) trained by
                     scripts/build_gate_dataset.py on UroLens + OpenUrine vs.
                     COCO. Its threshold is set so >=99.9% of real test slides
                     pass.

There is intentionally no blur check: real slides are mostly empty background
and their Laplacian variance goes as low as ~3, so any blur cut-off that would
catch a blurry photo also rejects real slides.

Public API:
    check_input(image)  -- raises ImageValidationError if the image is rejected
    get_gate()          -- returns the singleton InputGate instance
    InputGate           -- class for mocking in unit tests
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from urolens_ai.utils.exceptions import ImageValidationError, InferenceError
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment with safe defaults
# ---------------------------------------------------------------------------

_GATE_WEIGHTS_PATH: str = os.environ.get(
    "GATE_WEIGHTS_PATH",
    "src/urolens_ai/models/gate/gate.pt",
)
_GATE_MIN_SLIDE_PROB: float = float(os.environ.get("GATE_MIN_SLIDE_PROB", "0.43"))

# Exposure limits on 0-255 luminance. See module docstring for where they come from.
_MIN_MEAN_LUMINANCE = 25.0
_MAX_MEAN_LUMINANCE = 245.0
_MAX_CLIPPED_FRACTION = 0.6

_SLIDE_CLASS = "slide"


class InputGate:
    """
    Loads the slide/not_slide classifier once and checks images against it.

    Parameters
    ----------
    model_path : str
        Path to the classifier weights (.pt).
    min_slide_prob : float
        Images whose slide probability is below this are rejected.
    """

    def __init__(self, model_path: str, min_slide_prob: float) -> None:
        self.model_path = model_path
        self.min_slide_prob = min_slide_prob
        self.model = self._load_model(model_path)

    def _load_model(self, path: str) -> YOLO:
        if not Path(path).exists():
            logger.error(
                "gate_load_failed",
                extra={"error_code": "MODEL_NOT_LOADED", "path": path},
            )
            raise InferenceError(
                code="MODEL_NOT_LOADED",
                message=f"Input gate weights file not found at '{path}'.",
            )
        try:
            return YOLO(path)
        except Exception as exc:
            logger.error(
                "gate_load_failed",
                extra={"error_code": "MODEL_NOT_LOADED", "detail": str(exc)},
            )
            raise InferenceError(
                code="MODEL_NOT_LOADED",
                message=f"Failed to load input gate weights from '{path}': {exc}",
            ) from exc

    def slide_probability(self, image: np.ndarray) -> float:
        """
        Return the classifier's probability that `image` is urine microscopy.

        `image` is the RGB float32 array from preprocessing.normalise().
        """
        # Ultralytics treats numpy input as BGR -- same flip as YOLOEngine.run().
        bgr = np.ascontiguousarray(image[..., ::-1]).astype(np.uint8)
        try:
            result = self.model.predict(source=bgr, verbose=False)[0]  # type: ignore[no-untyped-call]
        except Exception as exc:
            logger.error(
                "gate_failed",
                extra={"error_code": "INFERENCE_FAILED", "detail": str(exc)},
            )
            raise InferenceError(
                code="INFERENCE_FAILED",
                message=f"Input gate inference failed: {exc}",
            ) from exc
        slide_index = {name: i for i, name in result.names.items()}[_SLIDE_CLASS]
        return float(result.probs.data[slide_index].item())

    def check(self, image: np.ndarray) -> None:
        """
        Raise ImageValidationError if `image` should not reach the detector.

        Raises
        ------
        ImageValidationError
            code="IMAGE_EXPOSURE" for near-black or blown-out frames.
            code="NOT_MICROSCOPY" if the classifier says this is not a slide.
        """
        check_exposure(image)

        prob = self.slide_probability(image)
        logger.debug("gate_checked", extra={"slide_prob": round(prob, 4)})
        if prob < self.min_slide_prob:
            raise ImageValidationError(
                code="NOT_MICROSCOPY",
                message=(
                    "This does not look like a urine microscopy image "
                    f"(slide probability {prob:.2f}). Please retake the image."
                ),
            )


def check_exposure(image: np.ndarray) -> None:
    """Raise ImageValidationError(code="IMAGE_EXPOSURE") for unusable exposure."""
    luminance = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    mean = float(luminance.mean())
    dark = float((luminance <= 5).mean())
    bright = float((luminance >= 250).mean())
    if (
        mean < _MIN_MEAN_LUMINANCE
        or mean > _MAX_MEAN_LUMINANCE
        or dark > _MAX_CLIPPED_FRACTION
        or bright > _MAX_CLIPPED_FRACTION
    ):
        raise ImageValidationError(
            code="IMAGE_EXPOSURE",
            message=(
                "The image is too dark or too bright to analyse "
                f"(mean brightness {mean:.0f}/255). Check the microscope light "
                "and retake the image."
            ),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_gate: InputGate | None = None


def get_gate() -> InputGate:
    """Return the singleton InputGate instance, loading it on first call."""
    global _gate
    if _gate is None:
        _gate = InputGate(
            model_path=_GATE_WEIGHTS_PATH,
            min_slide_prob=_GATE_MIN_SLIDE_PROB,
        )
    return _gate


def reset_gate() -> None:
    """Reset the singleton gate. For testing purposes only."""
    global _gate
    _gate = None


def check_input(image: np.ndarray) -> None:
    """Run all gate checks on a normalised RGB image. See InputGate.check()."""
    get_gate().check(image)
