"""
scripts/generate_test_fixtures.py
----------------------------------
Generates test fixture images from real microscopy samples.
Run once to populate tests/fixtures/sample_images/.
"""

from pathlib import Path
from PIL import Image

FIXTURES_DIR = Path("tests/fixtures/sample_images")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_IMAGES = [
    "01_191110_13_51_24_QC_LOW_03_bmp_jpg.rf.ddfe6d3079ce1c9db76062d21600f583.jpg",
    "02_191217_11_50_41_QC_HIGH_05_bmp_jpg.rf.50c1c2a28c5583cc9e20fb50ae719d16.jpg",
]

# Use first image as base
source = Image.open(SOURCE_IMAGES[0]).convert("RGB")

# valid_urine_640x480.jpg
source.resize((640, 480)).save(FIXTURES_DIR / "valid_urine_640x480.jpg", format="JPEG")
print("Created valid_urine_640x480.jpg")

# valid_urine_1280x960.jpg
source.resize((1280, 960)).save(FIXTURES_DIR / "valid_urine_1280x960.jpg", format="JPEG")
print("Created valid_urine_1280x960.jpg")

# too_small_300x200.jpg
source.resize((300, 200)).save(FIXTURES_DIR / "too_small_300x200.jpg", format="JPEG")
print("Created too_small_300x200.jpg")

# invalid.bmp
source.resize((640, 480)).save(FIXTURES_DIR / "invalid.bmp", format="BMP")
print("Created invalid.bmp")

print("Done.")