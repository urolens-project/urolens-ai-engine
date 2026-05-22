"""
tests/conftest.py
-----------------
Shared pytest fixtures for the urolens-ai-engine test suite.

Fixture images are generated programmatically using Pillow so no real
microscopy images are required for unit tests. Real images are only needed
for integration tests in STORY-AI-05 (Sprint 6).

Fixture categories:
    Image bytes   — valid/invalid images as raw bytes for preprocessing tests
    Classifications — sample particle classification dicts for rule engine tests
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_IMAGES_DIR = FIXTURES_DIR / "sample_images"
SAMPLE_CLASSIFICATIONS_DIR = FIXTURES_DIR / "sample_classifications"


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def _make_jpeg_bytes(width: int, height: int, include_exif: bool = False) -> bytes:
    """Create a minimal valid JPEG image of the given dimensions."""
    image = Image.new("RGB", (width, height), color=(128, 64, 32))
    buffer = io.BytesIO()
    if include_exif:
        # Embed minimal EXIF data (Orientation tag = 1)
        exif_data = image.getexif()
        exif_data[274] = 1  # Orientation tag
        image.save(buffer, format="JPEG", exif=exif_data.tobytes())
    else:
        image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


def _make_png_bytes(width: int, height: int, with_alpha: bool = False) -> bytes:
    """Create a minimal valid PNG image of the given dimensions."""
    mode = "RGBA" if with_alpha else "RGB"
    color = (128, 64, 32, 200) if with_alpha else (128, 64, 32)
    image = Image.new(mode, (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def _make_bmp_bytes(width: int, height: int) -> bytes:
    """Create a minimal valid BMP image — used to test format rejection."""
    image = Image.new("RGB", (width, height), color=(128, 64, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="BMP")
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Image byte fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_jpeg_640x480() -> bytes:
    """Valid JPEG image at exactly the minimum resolution (640 × 480)."""
    return _make_jpeg_bytes(640, 480)


@pytest.fixture
def valid_jpeg_1280x960() -> bytes:
    """Valid JPEG image above the minimum resolution (1280 × 960)."""
    return _make_jpeg_bytes(1280, 960)


@pytest.fixture
def valid_png_640x480() -> bytes:
    """Valid PNG image at exactly the minimum resolution (640 × 480)."""
    return _make_png_bytes(640, 480)


@pytest.fixture
def valid_png_with_alpha() -> bytes:
    """Valid RGBA PNG — normalise() must convert to RGB and drop alpha."""
    return _make_png_bytes(640, 480, with_alpha=True)


@pytest.fixture
def jpeg_with_exif() -> bytes:
    """Valid JPEG image containing EXIF metadata — must be stripped by normalise()."""
    return _make_jpeg_bytes(640, 480, include_exif=True)


@pytest.fixture
def too_small_jpeg() -> bytes:
    """JPEG image below minimum resolution (300 × 200)."""
    return _make_jpeg_bytes(300, 200)


@pytest.fixture
def too_small_width_only() -> bytes:
    """JPEG image with valid height but width below minimum (400 × 480)."""
    return _make_jpeg_bytes(400, 480)


@pytest.fixture
def too_small_height_only() -> bytes:
    """JPEG image with valid width but height below minimum (640 × 300)."""
    return _make_jpeg_bytes(640, 300)


@pytest.fixture
def bmp_image() -> bytes:
    """BMP image — must be rejected with FORMAT_UNSUPPORTED."""
    return _make_bmp_bytes(640, 480)


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Random bytes that cannot be decoded as any image format."""
    return b"\x00\x01\x02\x03\xff\xfe\xfd\xfc" * 100


@pytest.fixture
def empty_bytes() -> bytes:
    """Empty byte string — must be rejected as corrupt."""
    return b""


# ---------------------------------------------------------------------------
# Classification dict fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_classification() -> dict[str, int]:
    """All particle counts within normal ranges — expects all LOW, no_significant_indicators=True."""
    return {
        "bacteria": 0,
        "crystals": 1,
        "epithelial_cells": 2,
        "erythrocytes": 1,
        "leukocytes": 2,
        "mucus_threads": 0,
        "sperm_cells": 0,
        "trichomonas_vaginalis": 0,
        "urinary_casts": 0,
        "yeast": 0,
    }


@pytest.fixture
def high_uric_acid_classification() -> dict[str, int]:
    """Elevated uric acid crystals — expects Gout HIGH."""
    return {
        "uric_acid_crystals": 12,
        "erythrocytes": 1,
        "leukocytes": 2,
    }


@pytest.fixture
def rbc_casts_classification() -> dict[str, int]:
    """RBC casts present — expects Glomerulonephritis HIGH."""
    return {
        "rbc_casts": 3,
        "dysmorphic_rbc": 5,
        "erythrocytes": 8,
    }


@pytest.fixture
def calcium_oxalate_classification() -> dict[str, int]:
    """Elevated calcium oxalate — expects Nephrolithiasis MODERATE or HIGH."""
    return {
        "calcium_oxalate": 10,
        "erythrocytes": 2,
    }


@pytest.fixture
def empty_classification() -> dict[str, int]:
    """Empty classification dict — all counts default to 0."""
    return {}