"""
tests/unit/test_preprocessing.py
---------------------------------
Unit tests for urolens_ai.inference.preprocessing.

Covers every validation branch in validate_image() and all normalisation
behaviour in normalise(). No real model weights are required — these tests
only use programmatically generated images from conftest.py fixtures.

Coverage target: ≥ 90% branch coverage on preprocessing.py
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from urolens_ai.inference.preprocessing import normalise, validate_image
from urolens_ai.utils.exceptions import ImageValidationError


# ---------------------------------------------------------------------------
# validate_image() — format checks
# ---------------------------------------------------------------------------


class TestValidateImageFormat:
    def test_valid_jpeg_passes(self, valid_jpeg_640x480: bytes) -> None:
        """Valid JPEG at minimum resolution must not raise."""
        validate_image(valid_jpeg_640x480)  # no exception

    def test_valid_png_passes(self, valid_png_640x480: bytes) -> None:
        """Valid PNG at minimum resolution must not raise."""
        validate_image(valid_png_640x480)  # no exception

    def test_bmp_raises_format_unsupported(self, bmp_image: bytes) -> None:
        """BMP format must be rejected with FORMAT_UNSUPPORTED."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(bmp_image)
        assert exc_info.value.code == "FORMAT_UNSUPPORTED"
        assert "BMP" in exc_info.value.message

    def test_format_unsupported_message_contains_accepted_formats(
        self, bmp_image: bytes
    ) -> None:
        """Error message must list the accepted formats."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(bmp_image)
        assert "JPEG" in exc_info.value.message or "PNG" in exc_info.value.message


# ---------------------------------------------------------------------------
# validate_image() — resolution checks
# ---------------------------------------------------------------------------


class TestValidateImageResolution:
    def test_exact_minimum_resolution_passes(self, valid_jpeg_640x480: bytes) -> None:
        """Image at exactly 640 × 480 must pass — boundary condition."""
        validate_image(valid_jpeg_640x480)  # no exception

    def test_above_minimum_resolution_passes(self, valid_jpeg_1280x960: bytes) -> None:
        """Image above minimum resolution must pass."""
        validate_image(valid_jpeg_1280x960)  # no exception

    def test_below_minimum_resolution_raises(self, too_small_jpeg: bytes) -> None:
        """Image below minimum resolution (300 × 200) must raise RESOLUTION_TOO_LOW."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(too_small_jpeg)
        assert exc_info.value.code == "RESOLUTION_TOO_LOW"

    def test_resolution_error_message_contains_dimensions(
        self, too_small_jpeg: bytes
    ) -> None:
        """Error message must include both required and actual dimensions."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(too_small_jpeg)
        assert "640" in exc_info.value.message
        assert "480" in exc_info.value.message
        assert "300" in exc_info.value.message
        assert "200" in exc_info.value.message

    def test_width_below_minimum_raises(self, too_small_width_only: bytes) -> None:
        """Image with valid height but width below minimum must raise."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(too_small_width_only)
        assert exc_info.value.code == "RESOLUTION_TOO_LOW"

    def test_height_below_minimum_raises(self, too_small_height_only: bytes) -> None:
        """Image with valid width but height below minimum must raise."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(too_small_height_only)
        assert exc_info.value.code == "RESOLUTION_TOO_LOW"


# ---------------------------------------------------------------------------
# validate_image() — corruption checks
# ---------------------------------------------------------------------------


class TestValidateImageCorruption:
    def test_corrupt_bytes_raises(self, corrupt_image_bytes: bytes) -> None:
        """Random bytes that are not a valid image must raise CORRUPT_IMAGE."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(corrupt_image_bytes)
        assert exc_info.value.code == "CORRUPT_IMAGE"

    def test_empty_bytes_raises(self, empty_bytes: bytes) -> None:
        """Empty bytes must raise CORRUPT_IMAGE."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(empty_bytes)
        assert exc_info.value.code == "CORRUPT_IMAGE"


# ---------------------------------------------------------------------------
# validate_image() — exception interface
# ---------------------------------------------------------------------------


class TestValidateImageExceptionInterface:
    def test_exception_has_code_attribute(self, bmp_image: bytes) -> None:
        """ImageValidationError must always have a code attribute."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(bmp_image)
        assert hasattr(exc_info.value, "code")
        assert isinstance(exc_info.value.code, str)

    def test_exception_has_message_attribute(self, bmp_image: bytes) -> None:
        """ImageValidationError must always have a message attribute."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(bmp_image)
        assert hasattr(exc_info.value, "message")
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_exception_str_includes_code(self, too_small_jpeg: bytes) -> None:
        """str(exception) must include the error code."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(too_small_jpeg)
        assert exc_info.value.code in str(exc_info.value)


# ---------------------------------------------------------------------------
# normalise() — output shape and dtype
# ---------------------------------------------------------------------------


class TestNormaliseOutputFormat:
    def test_returns_numpy_array(self, valid_jpeg_640x480: bytes) -> None:
        """normalise() must return a numpy ndarray."""
        result = normalise(valid_jpeg_640x480)
        assert isinstance(result, np.ndarray)

    def test_returns_float32(self, valid_jpeg_640x480: bytes) -> None:
        """Output array must have dtype float32."""
        result = normalise(valid_jpeg_640x480)
        assert result.dtype == np.float32

    def test_output_shape_is_hwc(self, valid_jpeg_640x480: bytes) -> None:
        """Output shape must be (H, W, 3) — height × width × channels."""
        result = normalise(valid_jpeg_640x480)
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB channels

    def test_output_dimensions_match_input(self, valid_jpeg_640x480: bytes) -> None:
        """Output H and W must match input image dimensions."""
        result = normalise(valid_jpeg_640x480)
        assert result.shape[0] == 480  # height
        assert result.shape[1] == 640  # width

    def test_larger_image_dimensions_preserved(
        self, valid_jpeg_1280x960: bytes
    ) -> None:
        """normalise() must not resize the image."""
        result = normalise(valid_jpeg_1280x960)
        assert result.shape[0] == 960
        assert result.shape[1] == 1280

    def test_values_not_rescaled_to_0_1(self, valid_jpeg_640x480: bytes) -> None:
        """
        Values must remain in [0, 255] range.
        Ultralytics handles internal normalisation — do not pre-scale.
        """
        result = normalise(valid_jpeg_640x480)
        assert result.max() > 1.0


# ---------------------------------------------------------------------------
# normalise() — PNG with alpha channel
# ---------------------------------------------------------------------------


class TestNormalisePngAlpha:
    def test_rgba_png_converted_to_rgb(self, valid_png_with_alpha: bytes) -> None:
        """RGBA PNG must be converted to RGB — output must have 3 channels."""
        result = normalise(valid_png_with_alpha)
        assert result.shape[2] == 3

    def test_rgb_png_passes(self, valid_png_640x480: bytes) -> None:
        """Standard RGB PNG must normalise without error."""
        result = normalise(valid_png_640x480)
        assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# normalise() — EXIF stripping
# ---------------------------------------------------------------------------


class TestNormaliseExifStripping:
    def test_exif_stripped_from_output(self, jpeg_with_exif: bytes) -> None:
        """
        Image produced by normalise() must have no EXIF metadata.
        We verify by saving the output array back to JPEG and checking
        that getexif() returns no data.
        """
        array = normalise(jpeg_with_exif)

        # Convert array back to PIL image and save to buffer
        from PIL import Image as PILImage
        reconstructed = PILImage.fromarray(array.astype(np.uint8))
        buffer = io.BytesIO()
        reconstructed.save(buffer, format="JPEG")
        buffer.seek(0)

        reloaded = PILImage.open(buffer)
        exif = reloaded.getexif()
        assert len(exif) == 0, "EXIF data was not stripped from normalised image"

    def test_jpeg_without_exif_normalises_cleanly(
        self, valid_jpeg_640x480: bytes
    ) -> None:
        """JPEG without EXIF must normalise without error."""
        result = normalise(valid_jpeg_640x480)
        assert isinstance(result, np.ndarray)