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

import io
import os
from pathlib import Path

import pytest
from PIL import Image

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


# ---------------------------------------------------------------------------
# Boxes and image size for MedTech box review
# ---------------------------------------------------------------------------


def _with_exif_orientation(image_bytes: bytes, orientation: int) -> bytes:
    """Re-save a JPEG with an EXIF orientation tag, as a sideways phone photo has."""
    image = Image.open(io.BytesIO(image_bytes))
    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif, quality=95)
    return buffer.getvalue()


@pytest.fixture
def slide_image_bytes() -> bytes:
    """A real 640 x 480 test-split slide with 7 leukocytes, so boxes are non-empty."""
    path = Path("tests/fixtures/sample_images/slide_leukocytes_640x480.jpg")
    return path.read_bytes()


class TestInferDetections:
    def test_one_box_per_counted_particle(self, slide_image_bytes: bytes) -> None:
        result = infer(slide_image_bytes)
        assert result.detections
        assert len(result.detections) == result.filtered_detection_count
        for name, count in result.particles.items():
            assert sum(d.class_name == name for d in result.detections) == count

    def test_boxes_lie_inside_the_image(self, slide_image_bytes: bytes) -> None:
        result = infer(slide_image_bytes)
        assert result.detections
        for d in result.detections:
            x1, y1, x2, y2 = d.bbox
            assert 0 <= x1 < x2 <= result.image_width + 1
            assert 0 <= y1 < y2 <= result.image_height + 1

    def test_rotated_photo_reports_upright_size(self, slide_image_bytes: bytes) -> None:
        """
        EXIF orientation 6 stores a 640 x 480 image that displays as 480 x 640.
        The boxes are in upright pixels, so the size must be too: before the
        fix the size came back 640 x 480 and boxes ran past the bottom edge.
        """
        result = infer(_with_exif_orientation(slide_image_bytes, 6))
        assert (result.image_width, result.image_height) == (480, 640)
        assert result.detections
        for d in result.detections:
            assert d.bbox[2] <= result.image_width + 1
            assert d.bbox[3] <= result.image_height + 1
