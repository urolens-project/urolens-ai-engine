"""
tests/unit/test_input_gate.py
-----------------------------
Unit tests for urolens_ai.inference.input_gate.

The real classifier is never loaded — the YOLO class is mocked in every test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from urolens_ai.inference.input_gate import InputGate, check_exposure, get_gate, reset_gate
from urolens_ai.utils.exceptions import ImageValidationError, InferenceError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(slide_prob: float) -> MagicMock:
    """Mock classifier returning `slide_prob` for the 'slide' class."""
    result = MagicMock()
    result.names = {0: "not_slide", 1: "slide"}
    result.probs.data = torch.tensor([1.0 - slide_prob, slide_prob])
    model = MagicMock()
    model.predict.return_value = [result]
    return model


def _make_gate(mock_model: MagicMock, min_slide_prob: float = 0.5) -> InputGate:
    with patch("urolens_ai.inference.input_gate.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        with patch("urolens_ai.inference.input_gate.YOLO", return_value=mock_model):
            return InputGate(model_path="fake/gate.pt", min_slide_prob=min_slide_prob)


def _image(value: float) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


class TestCheckExposure:
    @pytest.mark.parametrize("value", [45.0, 160.0, 232.0])
    def test_real_slide_brightness_range_passes(self, value: float) -> None:
        """Real slides span mean luminance 43-233; those must pass."""
        check_exposure(_image(value))

    @pytest.mark.parametrize("value", [0.0, 10.0, 255.0])
    def test_black_or_blown_out_frame_rejected(self, value: float) -> None:
        with pytest.raises(ImageValidationError) as exc_info:
            check_exposure(_image(value))
        assert exc_info.value.code == "IMAGE_EXPOSURE"

    def test_eyepiece_vignette_passes(self) -> None:
        """OpenUrine eyepiece images have up to ~40% black border; must pass."""
        image = _image(170.0)
        image[:, :256] = 0.0  # 40% of columns black
        check_exposure(image)

    def test_mostly_black_frame_rejected(self) -> None:
        image = _image(200.0)
        image[:, :448] = 0.0  # 70% black
        with pytest.raises(ImageValidationError) as exc_info:
            check_exposure(image)
        assert exc_info.value.code == "IMAGE_EXPOSURE"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestInputGateCheck:
    def test_slide_passes(self) -> None:
        _make_gate(_make_mock_model(0.98)).check(_image(160.0))

    def test_non_slide_rejected(self) -> None:
        gate = _make_gate(_make_mock_model(0.03))
        with pytest.raises(ImageValidationError) as exc_info:
            gate.check(_image(160.0))
        assert exc_info.value.code == "NOT_MICROSCOPY"

    def test_threshold_is_respected(self) -> None:
        model = _make_mock_model(0.3)
        _make_gate(model, min_slide_prob=0.2).check(_image(160.0))
        with pytest.raises(ImageValidationError):
            _make_gate(model, min_slide_prob=0.4).check(_image(160.0))

    def test_exposure_checked_before_classifier(self) -> None:
        model = _make_mock_model(0.98)
        with pytest.raises(ImageValidationError) as exc_info:
            _make_gate(model).check(_image(0.0))
        assert exc_info.value.code == "IMAGE_EXPOSURE"
        model.predict.assert_not_called()

    def test_rgb_input_is_passed_to_model_as_bgr_uint8(self) -> None:
        model = _make_mock_model(0.98)
        image = _image(100.0)
        image[..., 0] = 200.0  # red

        _make_gate(model).check(image)

        sent = model.predict.call_args.kwargs["source"]
        assert sent.dtype == np.uint8
        assert sent[..., 2].min() == 200
        assert sent[..., 0].max() == 100

    def test_model_failure_raises_inference_error(self) -> None:
        model = _make_mock_model(0.98)
        model.predict.side_effect = RuntimeError("boom")
        with pytest.raises(InferenceError) as exc_info:
            _make_gate(model).check(_image(160.0))
        assert exc_info.value.code == "INFERENCE_FAILED"


# ---------------------------------------------------------------------------
# Loading / singleton
# ---------------------------------------------------------------------------


class TestGetGate:
    def test_missing_weights_raises(self) -> None:
        reset_gate()
        with patch("urolens_ai.inference.input_gate.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(InferenceError) as exc_info:
                get_gate()
        assert exc_info.value.code == "MODEL_NOT_LOADED"
        reset_gate()

    def test_returns_same_instance(self) -> None:
        reset_gate()
        with patch("urolens_ai.inference.input_gate.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("urolens_ai.inference.input_gate.YOLO", return_value=MagicMock()):
                assert get_gate() is get_gate()
        reset_gate()
