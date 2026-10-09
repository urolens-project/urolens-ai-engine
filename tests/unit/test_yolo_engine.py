"""
tests/unit/test_yolo_engine.py
------------------------------
Unit tests for urolens_ai.inference.yolo_engine.

The real YOLOv8 model is never loaded in unit tests — the YOLO class is
mocked in every test. This keeps unit tests fast and GPU-free.

Coverage target: ≥ 80% branch coverage on yolo_engine.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from urolens_ai.inference.yolo_engine import (
    Detection,
    YOLOEngine,
    get_engine,
    load_class_thresholds,
    reset_engine,
)
from urolens_ai.utils.exceptions import InferenceError


# ---------------------------------------------------------------------------
# Helpers — build mock YOLO results
# ---------------------------------------------------------------------------


def _make_mock_box(
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 50.0, 60.0),
) -> MagicMock:
    """Create a mock Ultralytics box object."""
    box = MagicMock()
    box.cls = [MagicMock()]
    box.cls[0].item.return_value = class_id
    box.conf = [MagicMock()]
    box.conf[0].item.return_value = confidence
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = list(bbox)
    return box


def _make_mock_result(
    boxes: list[MagicMock],
    names: dict[int, str],
) -> MagicMock:
    """Create a mock Ultralytics result object."""
    result = MagicMock()
    result.boxes = boxes
    result.names = names
    return result


def _make_engine_with_mock_model(mock_model: MagicMock) -> YOLOEngine:
    """Create a YOLOEngine with a pre-loaded mock model, bypassing _load_model."""
    with patch("urolens_ai.inference.yolo_engine.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        with patch("urolens_ai.inference.yolo_engine.YOLO", return_value=mock_model):
            engine = YOLOEngine(
                model_path="fake/weights.pt",
                conf=0.45,
                iou=0.5,
            )
    return engine


# ---------------------------------------------------------------------------
# YOLOEngine — model loading
# ---------------------------------------------------------------------------


class TestYOLOEngineLoading:
    def test_load_failure_missing_file_raises(self) -> None:
        """Missing weights file must raise InferenceError(code='MODEL_NOT_LOADED')."""
        with patch("urolens_ai.inference.yolo_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(InferenceError) as exc_info:
                YOLOEngine(model_path="nonexistent/weights.pt", conf=0.45, iou=0.5)
        assert exc_info.value.code == "MODEL_NOT_LOADED"

    def test_load_failure_ultralytics_error_raises(self) -> None:
        """Ultralytics error during load must raise InferenceError(code='MODEL_NOT_LOADED')."""
        with patch("urolens_ai.inference.yolo_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch(
                "urolens_ai.inference.yolo_engine.YOLO",
                side_effect=RuntimeError("corrupt weights"),
            ):
                with pytest.raises(InferenceError) as exc_info:
                    YOLOEngine(model_path="fake/weights.pt", conf=0.45, iou=0.5)
        assert exc_info.value.code == "MODEL_NOT_LOADED"
        assert "corrupt weights" in exc_info.value.message

    def test_successful_load(self) -> None:
        """Engine must load without error when weights file exists."""
        mock_model = MagicMock()
        with patch("urolens_ai.inference.yolo_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("urolens_ai.inference.yolo_engine.YOLO", return_value=mock_model):
                engine = YOLOEngine(model_path="fake/weights.pt", conf=0.45, iou=0.5)
        assert engine.model is mock_model


# ---------------------------------------------------------------------------
# YOLOEngine — inference
# ---------------------------------------------------------------------------


class TestYOLOEngineRun:
    def test_returns_list_of_detections(self) -> None:
        """run() must return a list of Detection objects."""
        mock_model = MagicMock()
        names = {0: "erythrocytes", 1: "leukocytes"}
        boxes = [_make_mock_box(0, "erythrocytes", 0.85)]
        mock_model.predict.return_value = [_make_mock_result(boxes, names)]

        engine = _make_engine_with_mock_model(mock_model)
        result = engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Detection)

    def test_detection_fields_mapped_correctly(self) -> None:
        """Detection fields must match the mock box values."""
        mock_model = MagicMock()
        names = {0: "erythrocytes"}
        boxes = [_make_mock_box(0, "erythrocytes", 0.85, (10.0, 20.0, 50.0, 60.0))]
        mock_model.predict.return_value = [_make_mock_result(boxes, names)]

        engine = _make_engine_with_mock_model(mock_model)
        detections = engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        d = detections[0]
        assert d.class_id == 0
        assert d.class_name == "erythrocytes"
        assert d.confidence == pytest.approx(0.85) # type: ignore
        assert d.bbox == (10.0, 20.0, 50.0, 60.0)

    def test_multiple_detections_returned(self) -> None:
        """run() must return all detections above the confidence threshold."""
        mock_model = MagicMock()
        names = {0: "erythrocytes", 1: "leukocytes"}
        boxes = [
            _make_mock_box(0, "erythrocytes", 0.85),
            _make_mock_box(0, "erythrocytes", 0.72),
            _make_mock_box(1, "leukocytes", 0.91),
        ]
        mock_model.predict.return_value = [_make_mock_result(boxes, names)]

        engine = _make_engine_with_mock_model(mock_model)
        detections = engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        assert len(detections) == 3

    def test_no_detections_returns_empty_list(self) -> None:
        """run() must return an empty list when no particles are detected."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_mock_result([], {})]

        engine = _make_engine_with_mock_model(mock_model)
        detections = engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        assert detections == []

    def test_none_boxes_returns_empty_list(self) -> None:
        """run() must handle result.boxes=None without error."""
        mock_model = MagicMock()
        result = MagicMock()
        result.boxes = None
        mock_model.predict.return_value = [result]

        engine = _make_engine_with_mock_model(mock_model)
        detections = engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        assert detections == []

    def test_inference_runtime_error_raises(self) -> None:
        """Runtime error during predict() must raise InferenceError(code='INFERENCE_FAILED')."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA out of memory")

        engine = _make_engine_with_mock_model(mock_model)
        with pytest.raises(InferenceError) as exc_info:
            engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        assert exc_info.value.code == "INFERENCE_FAILED"
        assert "CUDA out of memory" in exc_info.value.message

    def test_predict_called_with_correct_thresholds(self) -> None:
        """run() must pass conf and iou thresholds to model.predict()."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_mock_result([], {})]

        engine = _make_engine_with_mock_model(mock_model)
        engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        call_kwargs = mock_model.predict.call_args.kwargs
        assert call_kwargs["conf"] == 0.45
        assert call_kwargs["iou"] == 0.5

    def test_per_class_thresholds_filter_detections(self) -> None:
        """Each class is cut at its own threshold; unlisted classes use the default."""
        mock_model = MagicMock()
        names = {0: "bacteria", 1: "epithelial-cells", 2: "erythrocytes"}
        boxes = [
            _make_mock_box(0, "bacteria", 0.20),          # >= 0.15 -> kept
            _make_mock_box(1, "epithelial-cells", 0.50),  # <  0.55 -> dropped
            _make_mock_box(1, "epithelial-cells", 0.60),  # >= 0.55 -> kept
            _make_mock_box(2, "erythrocytes", 0.30),      # <  0.35 default -> dropped
            _make_mock_box(2, "erythrocytes", 0.40),      # >= 0.35 default -> kept
        ]
        mock_model.predict.return_value = [_make_mock_result(boxes, names)]

        with patch("urolens_ai.inference.yolo_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("urolens_ai.inference.yolo_engine.YOLO", return_value=mock_model):
                engine = YOLOEngine(
                    model_path="fake/weights.pt",
                    conf=0.15,
                    iou=0.5,
                    class_thresholds={"bacteria": 0.15, "epithelial-cells": 0.55},
                    default_conf=0.35,
                )
        detections = engine.run(np.zeros((480, 640, 3), dtype=np.float32))

        kept = sorted((d.class_name, d.confidence) for d in detections)
        assert kept == [("bacteria", 0.20), ("epithelial-cells", 0.60), ("erythrocytes", 0.40)]

    def test_rgb_input_is_passed_to_model_as_bgr(self) -> None:
        """
        normalise() produces RGB, but Ultralytics treats numpy input as BGR.

        Passing RGB straight through swaps red and blue for the model and cost
        about 7 points of recall on the test split.
        """
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_mock_result([], {})]
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        rgb[..., 0] = 200.0  # red channel only

        engine = _make_engine_with_mock_model(mock_model)
        engine.run(rgb)

        sent = mock_model.predict.call_args.kwargs["source"]
        assert sent[..., 2].min() == 200.0  # red now in BGR position 2
        assert sent[..., 0].max() == 0.0
        assert sent.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# get_engine() singleton
# ---------------------------------------------------------------------------


class TestGetEngine:
    def test_returns_same_instance_on_repeated_calls(self) -> None:
        """get_engine() must return the same instance every time."""

        mock_model = MagicMock()
        # Reset singleton for test isolation
        reset_engine()

        # load_class_thresholds is stubbed because the Path mock would hand
        # yaml.safe_load a MagicMock "file" that never hits EOF and eats all RAM.
        with patch("urolens_ai.inference.yolo_engine.Path") as mock_path, patch(
            "urolens_ai.inference.yolo_engine.load_class_thresholds", return_value={}
        ):
            mock_path.return_value.exists.return_value = True
            with patch(
                "urolens_ai.inference.yolo_engine.YOLO", return_value=mock_model
            ):
                first = get_engine()
                second = get_engine()

        assert first is second

        # Clean up singleton after test
        reset_engine()

    def test_missing_weights_raises_on_first_call(self) -> None:
        """get_engine() must raise InferenceError if weights are missing."""

        reset_engine()  # Ensure singleton is reset before test

        with patch("urolens_ai.inference.yolo_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(InferenceError) as exc_info:
                get_engine()

        assert exc_info.value.code == "MODEL_NOT_LOADED"
        reset_engine()  # Clean up singleton after test


# ---------------------------------------------------------------------------
# load_class_thresholds()
# ---------------------------------------------------------------------------


class TestLoadClassThresholds:
    def test_reads_per_class_mapping(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "thresholds.yaml"
        path.write_text("per_class:\n  bacteria: 0.15\n  sperm-cells: 0.3\n", encoding="utf-8")
        assert load_class_thresholds(str(path)) == {"bacteria": 0.15, "sperm-cells": 0.3}

    def test_missing_file_falls_back_to_uniform(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert load_class_thresholds(str(tmp_path / "nope.yaml")) == {}

    def test_shipped_file_covers_every_model_class(self) -> None:
        """The shipped thresholds must name real classes, or they silently do nothing."""
        shipped = load_class_thresholds("src/urolens_ai/models/yolov8/thresholds.yaml")
        model_classes = {
            "bacteria", "crystals", "epithelial-cells", "erythrocytes", "leukocytes",
            "mucus-threads", "sperm-cells", "trichomonas-vaginalis", "urinary-casts", "yeast",
        }
        assert set(shipped) == model_classes
        assert all(0.0 < v < 1.0 for v in shipped.values())