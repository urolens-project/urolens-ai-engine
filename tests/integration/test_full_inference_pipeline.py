"""
tests/integration/test_full_inference_pipeline.py
--------------------------------------------------
Integration tests for the full infer() pipeline.

These tests require real model weights at MODEL_WEIGHTS_PATH and real
fixture images in tests/fixtures/sample_images/. They are slower than
unit tests and should be run separately.

Run with:
    pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from urolens_ai import infer
from urolens_ai.schemas.inference import InferenceResult
from urolens_ai.utils.exceptions import ImageValidationError

# ---------------------------------------------------------------------------
# Skip entire module if model weights are not present
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "MODEL_WEIGHTS_PATH",
    "src/urolens_ai/models/yolov8/best.pt",
)

pytestmark = pytest.mark.integration


@pytest.fixture
def valid_image_bytes() -> bytes:
    path = Path("tests/fixtures/sample_images/valid_urine_640x480.jpg")
    return path.read_bytes()


@pytest.fixture
def large_image_bytes() -> bytes:
    path = Path("tests/fixtures/sample_images/valid_urine_1280x960.jpg")
    return path.read_bytes()


@pytest.fixture
def too_small_image_bytes() -> bytes:
    path = Path("tests/fixtures/sample_images/too_small_300x200.jpg")
    return path.read_bytes()


@pytest.fixture
def bmp_image_bytes() -> bytes:
    path = Path("tests/fixtures/sample_images/invalid.bmp")
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestInferHappyPath:
    def test_valid_image_returns_inference_result(
        self, valid_image_bytes: bytes
    ) -> None:
        """Valid image must return a fully populated InferenceResult."""
        result = infer(valid_image_bytes)
        assert isinstance(result, InferenceResult)

    def test_result_has_model_version(self, valid_image_bytes: bytes) -> None:
        """model_version must match MODEL_VERSION env var."""
        result = infer(valid_image_bytes)
        expected = os.environ.get("MODEL_VERSION", "unknown")
        assert result.model_version == expected

    def test_result_has_positive_inference_time(
        self, valid_image_bytes: bytes
    ) -> None:
        """inference_time_ms must be a positive float."""
        result = infer(valid_image_bytes)
        assert result.inference_time_ms > 0.0

    def test_result_image_dimensions_match_input(
        self, valid_image_bytes: bytes
    ) -> None:
        """image_width and image_height must match the input image dimensions."""
        result = infer(valid_image_bytes)
        assert result.image_width == 640
        assert result.image_height == 480

    def test_result_particles_is_dict(self, valid_image_bytes: bytes) -> None:
        """particles must be a dict mapping str to int."""
        result = infer(valid_image_bytes)
        assert isinstance(result.particles, dict)
        for k, v in result.particles.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    def test_result_confidence_scores_is_dict(self, valid_image_bytes: bytes) -> None:
        """confidence_scores must be a dict mapping str to float."""
        result = infer(valid_image_bytes)
        assert isinstance(result.confidence_scores, dict)
        for k, v in result.confidence_scores.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_confidence_keys_match_particle_keys(
        self, valid_image_bytes: bytes
    ) -> None:
        """confidence_scores keys must match particles keys."""
        result = infer(valid_image_bytes)
        assert set(result.particles.keys()) == set(result.confidence_scores.keys())

    def test_filtered_count_matches_particle_sum(
        self, valid_image_bytes: bytes
    ) -> None:
        """filtered_detection_count must equal sum of all particle counts."""
        result = infer(valid_image_bytes)
        assert result.filtered_detection_count == sum(result.particles.values())

    def test_raw_count_gte_filtered_count(self, valid_image_bytes: bytes) -> None:
        """raw_detection_count must be >= filtered_detection_count."""
        result = infer(valid_image_bytes)
        assert result.raw_detection_count >= result.filtered_detection_count

    def test_large_image_returns_correct_dimensions(
        self, large_image_bytes: bytes
    ) -> None:
        """1280 × 960 image must return correct dimensions."""
        result = infer(large_image_bytes)
        assert result.image_width == 1280
        assert result.image_height == 960


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


class TestInferValidationFailures:
    def test_too_small_image_raises_resolution_error(
        self, too_small_image_bytes: bytes
    ) -> None:
        """Image below 640 × 480 must raise ImageValidationError before inference."""
        with pytest.raises(ImageValidationError) as exc_info:
            infer(too_small_image_bytes)
        assert exc_info.value.code == "RESOLUTION_TOO_LOW"

    def test_bmp_image_raises_format_error(self, bmp_image_bytes: bytes) -> None:
        """BMP image must raise ImageValidationError before inference."""
        with pytest.raises(ImageValidationError) as exc_info:
            infer(bmp_image_bytes)
        assert exc_info.value.code == "FORMAT_UNSUPPORTED"

    def test_corrupt_bytes_raises_corrupt_error(self) -> None:
        """Corrupt bytes must raise ImageValidationError before inference."""
        with pytest.raises(ImageValidationError) as exc_info:
            infer(b"\x00\x01\x02\x03" * 50)
        assert exc_info.value.code == "CORRUPT_IMAGE"
    def test_non_microscopy_photo_raises_not_microscopy(self) -> None:
        """
        A normal photo must be rejected by the input gate, not run through the
        detector. Before the gate, a classroom selfie came back as "2 sperm cells".
        """
        image_bytes = Path("tests/fixtures/sample_images/not_microscopy_kitchen.jpg").read_bytes()
        with pytest.raises(ImageValidationError) as exc_info:
            infer(image_bytes)
        assert exc_info.value.code == "NOT_MICROSCOPY"
