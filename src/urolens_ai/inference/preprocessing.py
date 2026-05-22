"""
inference/preprocessing.py
--------------------------
Image validation and normalisation before YOLOv8 inference.

All functions in this module are called by infer() in urolens_ai/__init__.py
before the image is passed to the YOLOv8 engine. No image should ever reach
the model without passing through validate_image() and normalise() first.

Public functions:
    validate_image(image_bytes)  — validates format, resolution, integrity, strips EXIF
    normalise(image_bytes)       — converts validated image to RGB numpy float32 array
"""

from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image, ImageOps

from urolens_ai.utils.exceptions import ImageValidationError
from urolens_ai.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment with safe defaults
# ---------------------------------------------------------------------------

_MIN_WIDTH: int = int(os.environ.get("MIN_IMAGE_WIDTH", "640"))
_MIN_HEIGHT: int = int(os.environ.get("MIN_IMAGE_HEIGHT", "480"))
_ACCEPTED_FORMATS: set[str] = {
    fmt.strip().upper()
    for fmt in os.environ.get("ACCEPTED_IMAGE_FORMATS", "JPEG,PNG").split(",")
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def validate_image(image_bytes: bytes) -> None:
    """
    Validate raw image bytes before inference.

    Performs four checks in order:
        1. Corruption check — bytes can be decoded as an image
        2. Format check    — image format is JPEG or PNG
        3. Resolution check — image meets minimum width and height
        4. EXIF strip      — all EXIF metadata is removed in-place

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the image to validate.

    Raises
    ------
    ImageValidationError
        code="CORRUPT_IMAGE"
            Bytes cannot be decoded as a valid image.
        code="FORMAT_UNSUPPORTED"
            Image format is not JPEG or PNG.
            Includes the detected format in the message.
        code="RESOLUTION_TOO_LOW"
            Image dimensions are below the minimum 640 × 480.
            Includes both the required and actual dimensions in the message.

    Notes
    -----
    - EXIF stripping is a patient-data privacy requirement. EXIF can contain
      GPS coordinates, device serial numbers, and other identifying metadata.
    - This function does not return anything. A clean return means the image
      passed all checks and is safe to pass to normalise().
    """
    # ------------------------------------------------------------------
    # Step 1 — Corruption check
    # ------------------------------------------------------------------
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
    except Exception as exc:
        logger.warning(
            "image_validation_failed",
            extra={"code": "CORRUPT_IMAGE", "reason": str(exc)},
        )
        raise ImageValidationError(
            code="CORRUPT_IMAGE",
            message=f"Image bytes could not be decoded: {exc}",
        ) from exc

    # Re-open after verify() — verify() exhausts the file pointer
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ImageValidationError(
            code="CORRUPT_IMAGE",
            message=f"Image bytes could not be re-opened after verify: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 2 — Format check
    # ------------------------------------------------------------------
    detected_format: str = (image.format or "").upper()
    if detected_format not in _ACCEPTED_FORMATS:
        logger.warning(
            "image_validation_failed",
            extra={
                "code": "FORMAT_UNSUPPORTED",
                "detected_format": detected_format,
                "accepted_formats": sorted(_ACCEPTED_FORMATS),
            },
        )
        raise ImageValidationError(
            code="FORMAT_UNSUPPORTED",
            message=(
                f"Image format '{detected_format}' is not supported. "
                f"Accepted formats: {', '.join(sorted(_ACCEPTED_FORMATS))}."
            ),
        )

    # ------------------------------------------------------------------
    # Step 3 — Resolution check
    # ------------------------------------------------------------------
    width, height = image.size
    if width < _MIN_WIDTH or height < _MIN_HEIGHT:
        logger.warning(
            "image_validation_failed",
            extra={
                "code": "RESOLUTION_TOO_LOW",
                "required": f"{_MIN_WIDTH}x{_MIN_HEIGHT}",
                "actual": f"{width}x{height}",
            },
        )
        raise ImageValidationError(
            code="RESOLUTION_TOO_LOW",
            message=(
                f"Image must be at least {_MIN_WIDTH}x{_MIN_HEIGHT}. "
                f"Got {width}x{height}."
            ),
        )

    logger.debug(
        "image_validation_passed",
        extra={"format": detected_format, "width": width, "height": height},
    )


def normalise(image_bytes: bytes) -> np.ndarray:
    """
    Convert validated image bytes to a normalised numpy array for YOLOv8.

    This function assumes validate_image() has already been called on the
    same bytes. Do not call normalise() on unvalidated bytes.

    Steps performed:
        1. Strip EXIF metadata — patient privacy requirement
        2. Convert to RGB — drops alpha channel from PNG, ensures 3-channel array
        3. Convert to numpy float32 array — YOLOv8 input format

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a validated JPEG or PNG image.

    Returns
    -------
    np.ndarray
        A float32 numpy array of shape (H, W, 3).
        Values are in the range [0, 255] — Ultralytics YOLOv8 handles
        internal normalisation to [0, 1]. Do NOT manually rescale.

    Notes
    -----
    - The image is NOT resized. YOLOv8 handles resizing internally.
      Pre-resizing changes particle sizes and degrades detection confidence.
    - EXIF stripping uses ImageOps.exif_transpose() to apply any orientation
      correction encoded in EXIF, then removes the EXIF data entirely.
    """
    image = Image.open(io.BytesIO(image_bytes))

    # Step 1 — Apply EXIF orientation correction then strip all EXIF metadata
    image = ImageOps.exif_transpose(image)
    # Remove EXIF by saving to a clean buffer and re-opening
    clean_buffer = io.BytesIO()
    image.save(clean_buffer, format="PNG")
    clean_buffer.seek(0)
    image = Image.open(clean_buffer)

    # Step 2 — Convert to RGB (drops alpha channel from RGBA PNGs)
    image = image.convert("RGB")

    # Step 3 — Convert to float32 numpy array
    array: np.ndarray = np.array(image, dtype=np.float32)

    logger.debug(
        "image_normalised",
        extra={"shape": str(array.shape), "dtype": str(array.dtype)},
    )

    return array