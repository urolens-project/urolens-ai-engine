"""
inference/yolo_engine.py
------------------------
YOLOv8 model wrapper for UroLens particle detection.

Provides the YOLOEngine class which loads YOLOv8 weights once and runs
inference on preprocessed microscopy images. A module-level singleton
(get_engine()) ensures the model is loaded only once per process.

Public API:
    get_engine()  — returns the singleton YOLOEngine instance
    YOLOEngine    — class for mocking in unit tests
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from urolens_ai.utils.exceptions import InferenceError
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment with safe defaults
# ---------------------------------------------------------------------------

_MODEL_WEIGHTS_PATH: str = os.environ.get(
    "MODEL_WEIGHTS_PATH",
    "src/urolens_ai/models/yolov8/best.pt",
)
_INFERENCE_CONF_THRESHOLD: float = float(
    os.environ.get("INFERENCE_CONF_THRESHOLD", "0.35")
)
_INFERENCE_IOU_THRESHOLD: float = float(
    os.environ.get("INFERENCE_IOU_THRESHOLD", "0.5")
)
# Per-class overrides of INFERENCE_CONF_THRESHOLD, chosen by
# scripts/evaluate.py --sweep-per-class. Classes not listed use the global value.
_CLASS_THRESHOLDS_PATH: str = os.environ.get(
    "CLASS_THRESHOLDS_PATH",
    "src/urolens_ai/models/yolov8/thresholds.yaml",
)


def load_class_thresholds(path: str) -> dict[str, float]:
    """
    Read the per-class confidence thresholds file.

    Keys are YOLO class names as the model emits them (dashes, e.g.
    "urinary-casts"). A missing file is not an error: every class then uses
    INFERENCE_CONF_THRESHOLD, which is the pre-per-class behaviour.
    """
    if not Path(path).exists():
        logger.warning("class_thresholds_missing", extra={"path": path})
        return {}
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): float(v) for k, v in (config.get("per_class") or {}).items()}


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """
    A single bounding box detection from YOLOv8.

    Attributes
    ----------
    class_id : int
        Zero-based class index matching the order in labels.txt.
    class_name : str
        Human-readable class name from labels.txt.
    confidence : float
        Detection confidence score in [0, 1].
    bbox : tuple[float, float, float, float]
        Bounding box as (x1, y1, x2, y2) in pixel coordinates.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# YOLOEngine
# ---------------------------------------------------------------------------


class YOLOEngine:
    """
    Wrapper around the Ultralytics YOLOv8 model.

    Loads model weights once on initialisation and exposes a run() method
    that returns a list of Detection objects for a given image array.

    This class is designed to be instantiated once via get_engine() and
    reused for all inference calls. It is a class (not a module-level
    function) specifically to enable mocking in unit tests.

    Parameters
    ----------
    model_path : str
        Path to the YOLOv8 weights file (.pt).
    conf : float
        Confidence threshold passed to the model. Must be no higher than any
        per-class threshold, or those classes lose detections before filtering.
    iou : float
        IoU threshold for Non-Maximum Suppression (NMS).
        Applied automatically by Ultralytics.
    class_thresholds : dict[str, float] | None
        Per-class cut-offs keyed by YOLO class name, applied after the model.
    default_conf : float | None
        Cut-off for classes not in class_thresholds. Defaults to `conf`.
    """

    def __init__(
        self,
        model_path: str,
        conf: float,
        iou: float,
        class_thresholds: dict[str, float] | None = None,
        default_conf: float | None = None,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.class_thresholds = class_thresholds or {}
        self.default_conf = conf if default_conf is None else default_conf
        self.model = self._load_model(model_path)

    def _load_model(self, path: str) -> YOLO:
        """
        Load YOLOv8 weights from disk.

        Raises
        ------
        InferenceError
            code="MODEL_NOT_LOADED" if the weights file does not exist
            or cannot be loaded by Ultralytics.
        """
        if not Path(path).exists():
            logger.error(
                "model_load_failed",
                extra={"error_code": "MODEL_NOT_LOADED", "path": path},
            )
            raise InferenceError(
                code="MODEL_NOT_LOADED",
                message=f"Model weights file not found at '{path}'.",
            )

        try:
            logger.debug("model_loading", extra={"path": path})
            model = YOLO(path)
            logger.debug(
                "model_loaded",
                extra={"path": path},
            )
            return model
        except Exception as exc:
            logger.error(
                "model_load_failed",
                extra={"error_code": "MODEL_NOT_LOADED", "detail": str(exc)},
            )
            raise InferenceError(
                code="MODEL_NOT_LOADED",
                message=f"Failed to load model weights from '{path}': {exc}",
            ) from exc

    def run(self, image: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8 inference on a preprocessed image array.

        Parameters
        ----------
        image : np.ndarray
            Float32 array of shape (H, W, 3) produced by preprocessing.normalise().

        Returns
        -------
        list[Detection]
            All detections above the confidence threshold, after NMS.
            Empty list if no particles are detected.

        Raises
        ------
        InferenceError
            code="INFERENCE_FAILED" if the model raises during forward pass.
        """
        logger.debug("inference_started", extra={"model_version": os.environ.get("MODEL_VERSION", "unknown")})
        start = time.perf_counter()

        # normalise() returns RGB, but Ultralytics treats numpy input as BGR
        # (OpenCV convention). Without this flip the model sees red and blue
        # swapped, which cost ~7 points of recall on the test split.
        bgr = np.ascontiguousarray(image[..., ::-1])

        try:
            results = self.model.predict( # type: ignore[no-untyped-call]
                source=bgr,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as exc:
            logger.error(
                "inference_failed",
                extra={"error_code": "INFERENCE_FAILED", "detail": str(exc)},
            )
            raise InferenceError(
                code="INFERENCE_FAILED",
                message=f"YOLOv8 inference failed: {exc}",
            ) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("inference_completed", extra={"duration_ms": round(elapsed_ms, 2)})

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                confidence = float(box.conf[0].item())
                if confidence < self.class_thresholds.get(class_name, self.default_conf):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist() # type: ignore[no-untyped-call]
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2), # type: ignore[no-untyped-call]
                    )
                )

        return detections


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: YOLOEngine | None = None


def get_engine() -> YOLOEngine:
    """
    Return the singleton YOLOEngine instance, loading it on first call.

    The model is loaded once at first access and reused for all subsequent
    inference calls within the same process. This avoids the overhead of
    reloading weights on every request.

    Returns
    -------
    YOLOEngine
        The singleton engine instance.

    Raises
    ------
    InferenceError
        code="MODEL_NOT_LOADED" if the weights file is missing or unreadable.
    """
    global _engine
    if _engine is None:
        class_thresholds = load_class_thresholds(_CLASS_THRESHOLDS_PATH)
        _engine = YOLOEngine(
            model_path=_MODEL_WEIGHTS_PATH,
            # The model must keep everything down to the lowest per-class cut-off;
            # run() then applies each class's own threshold.
            conf=min([_INFERENCE_CONF_THRESHOLD, *class_thresholds.values()]),
            iou=_INFERENCE_IOU_THRESHOLD,
            class_thresholds=class_thresholds,
            default_conf=_INFERENCE_CONF_THRESHOLD,
        )
    return _engine

def reset_engine() -> None:
    """Reset the singleton engine. For testing purposes only."""
    global _engine
    _engine = None