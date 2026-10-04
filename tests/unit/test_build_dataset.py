"""
tests/unit/test_build_dataset.py
--------------------------------
Unit tests for scripts/build_dataset.py.

The regression these guard against is the one that invalidated every accuracy
number the project had produced: near-duplicate images -- including mirrored
copies -- landing in different splits. A synthetic corpus is built on disk with a
known flipped pair planted in it, so the flip-aware path is proven to fire rather
than assumed to.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_module():
    """Import build_dataset.py by path — scripts/ is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "build_dataset", SCRIPTS_DIR / "build_dataset.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_dataset"] = module
    spec.loader.exec_module(module)
    return module


bd = _load_module()


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------


def _textured_image(rng: np.random.Generator, size: int = 320) -> np.ndarray:
    """
    A field with enough particle-like structure to clear the texture floor.

    Blobs on a bright ground, mimicking sediment under brightfield illumination.
    """
    canvas = np.full((size, size), 235.0, dtype=np.float32)
    for _ in range(40):
        cy, cx = rng.integers(12, size - 12, size=2)
        radius = int(rng.integers(4, 9))
        y, x = np.ogrid[:size, :size]
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius**2
        canvas[mask] -= float(rng.integers(60, 130))
    return np.clip(canvas, 0, 255)


def _write(
    root: Path, split: str, name: str, array: np.ndarray, boxes: list[tuple]
) -> None:
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).convert("RGB").save(
        images / f"{name}_jpg.rf.{abs(hash(name + split)) % (16**8):08x}.jpg",
        quality=95,
    )
    stem = next(images.glob(f"{name}_jpg.rf.*")).stem
    (labels / f"{stem}.txt").write_text(
        "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes),
        encoding="utf-8",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """
    A miniature UroLens-3, carrying every defect the rebuild has to handle.

    Planted deliberately:
      * a horizontally mirrored copy of one field, under a DIFFERENT name,
        in a different split — the case filename dedup cannot catch
      * a same-name brightness variant across splits
      * a group whose copies disagree on box count, one of them blank
    """
    root = tmp_path / "src"
    rng = np.random.default_rng(1234)

    base = _textured_image(rng)
    boxes = [(1, 0.25, 0.40, 0.05, 0.06), (3, 0.70, 0.20, 0.04, 0.05)]

    # 1. The mirrored pair: same picture, different name, opposite splits.
    _write(root, "train", "20200101000001_01", base, boxes)
    mirrored = base[:, ::-1]
    mirrored_boxes = [(c, 1.0 - x, y, w, h) for c, x, y, w, h in boxes]
    _write(root, "test", "20200101000002_01", mirrored, mirrored_boxes)

    # 2. Same-name brightness variant straddling train/valid.
    second = _textured_image(rng)
    second_boxes = [(4, 0.5, 0.5, 0.05, 0.05)]
    _write(root, "train", "nh00001", second, second_boxes)
    _write(root, "valid", "nh00001", np.clip(second * 0.85, 0, 255), second_boxes)

    # 3. Label conflict: an annotated copy and an identical blank one.
    third = _textured_image(rng)
    third_boxes = [(9, 0.3, 0.3, 0.05, 0.05), (9, 0.6, 0.7, 0.05, 0.05)]
    _write(root, "train", "nh00002", third, third_boxes)
    _write(root, "valid", "nh00002", third, [])

    # 4. Filler so stratification has something to distribute.
    for i in range(24):
        split = ("train", "valid", "test")[i % 3]
        _write(
            root,
            split,
            f"filler{i:03d}",
            _textured_image(rng),
            [(i % 10, 0.5, 0.5, 0.05, 0.05)],
        )
    return root


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFilenameParsing:
    def test_source_basename_strips_roboflow_suffix(self) -> None:
        assert (
            bd.source_basename("00501_200205_02_34_09_-_01_bmp_jpg.rf.abc123.jpg")
            == "00501_200205_02_34_09_-_01_bmp"
        )

    @pytest.mark.parametrize(
        "name,family,groupable",
        [
            ("00501_200205_02_34_09_-_01_bmp", "field_bmp", True),
            ("01_191105_14_32_08_QC_LOW_03_bmp", "field_bmp", True),
            ("WIN_20211006_17_08_50_Pro", "win_camera", True),
            ("20211106002601_01", "timestamp_field", True),
            ("nh01021", "nh_nl_id", False),
            ("nl00306", "nh_nl_id", False),
            ("1580332483256", "pure_number", False),
            ("a1582149090150", "other", False),
        ],
    )
    def test_family_detection(self, name: str, family: str, groupable: bool) -> None:
        assert bd.family_of(name) == (family, groupable)

    def test_specimen_key_strips_field_number(self) -> None:
        assert (
            bd.specimen_key("00501_200205_02_34_09_-_01_bmp", True)
            == "00501_200205_02_34_09_-_01"
        )

    def test_specimen_key_is_none_without_provenance(self) -> None:
        """75% of the corpus encodes no patient id — that must not be faked."""
        assert bd.specimen_key("nh01021", False) is None


class TestBoxTransforms:
    def test_fliplr_mirrors_x_only(self) -> None:
        assert bd.transform_box((2, 0.25, 0.40, 0.1, 0.2), 1) == (2, 0.75, 0.40, 0.1, 0.2)

    def test_flipud_mirrors_y_only(self) -> None:
        assert bd.transform_box((2, 0.25, 0.40, 0.1, 0.2), 2) == (2, 0.25, 0.60, 0.1, 0.2)

    def test_rot180_mirrors_both(self) -> None:
        assert bd.transform_box((2, 0.25, 0.40, 0.1, 0.2), 3) == (2, 0.75, 0.60, 0.1, 0.2)

    def test_identity_is_unchanged(self) -> None:
        box = (2, 0.25, 0.40, 0.1, 0.2)
        assert bd.transform_box(box, 0) == box


class TestRepresentativeSelection:
    def test_richest_copy_wins(self) -> None:
        """A blank copy of an image annotated elsewhere is a miss, not an empty field."""
        recs = [
            bd.ImageRec(0, "train", "a.jpg", Path("a"), Path("a"), boxes=[]),
            bd.ImageRec(
                1, "valid", "b.jpg", Path("b"), Path("b"), boxes=[(0, 0.1, 0.1, 0.1, 0.1)]
            ),
        ]
        assert bd.choose_representative(recs, [0, 1]) == 1

    def test_tie_breaks_on_filename_for_reproducibility(self) -> None:
        recs = [
            bd.ImageRec(0, "train", "zzz.jpg", Path("z"), Path("z"), boxes=[]),
            bd.ImageRec(1, "valid", "aaa.jpg", Path("a"), Path("a"), boxes=[]),
        ]
        assert bd.choose_representative(recs, [0, 1]) == 0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestRebuild:
    def test_mirrored_duplicate_is_detected(self, corpus: Path) -> None:
        """The regression test that matters: a flip under a different name."""
        recs = bd.collect(corpus)
        descriptors = bd.fingerprint(recs, verbose=False)
        pairs = bd.find_visual_duplicates(recs, descriptors, verbose=False)

        mirrored = [p for p in pairs if p.transform != 0]
        assert mirrored, "flipped duplicate was not detected"
        names = {recs[p.i].basename for p in mirrored} | {
            recs[p.j].basename for p in mirrored
        }
        assert {"20200101000001_01", "20200101000002_01"} <= names

    def test_output_splits_are_disjoint(self, corpus: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out"
        assert bd.main(["--src", str(corpus), "--dst", str(dst), "--seed", "0"]) == 0

        manifest = json.loads((dst / "split_manifest.json").read_text())
        by_split: dict[str, set[str]] = {}
        for filename, meta in manifest["assignments"].items():
            by_split.setdefault(meta["split"], set()).add(bd.source_basename(filename))
        seen: set[str] = set()
        for names in by_split.values():
            assert not (seen & names), "a source name appears in two splits"
            seen |= names

    def test_duplicates_are_collapsed(self, corpus: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out"
        bd.main(["--src", str(corpus), "--dst", str(dst), "--seed", "0"])
        manifest = json.loads((dst / "split_manifest.json").read_text())
        assert manifest["kept_images"] < manifest["source_images"]
        assert manifest["mirrored_pairs"] >= 1

    def test_blank_copy_is_discarded_in_favour_of_annotated(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out"
        bd.main(["--src", str(corpus), "--dst", str(dst), "--seed", "0"])
        manifest = json.loads((dst / "split_manifest.json").read_text())
        kept = {
            bd.source_basename(name): meta["boxes"]
            for name, meta in manifest["assignments"].items()
        }
        assert kept.get("nh00002") == 2, "the blank copy was kept over the annotated one"

    def test_conflicts_are_reported(self, corpus: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out"
        bd.main(["--src", str(corpus), "--dst", str(dst), "--seed", "0"])
        rows = (dst / "conflicts.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) >= 2, "the box-count disagreement was not logged"
        assert any("nh00002" in row for row in rows[1:])

    def test_run_is_deterministic(self, corpus: Path, tmp_path: Path) -> None:
        """A rebuild that cannot be reproduced cannot be defended in a paper."""
        first, second = tmp_path / "a", tmp_path / "b"
        bd.main(["--src", str(corpus), "--dst", str(first), "--seed", "0"])
        bd.main(["--src", str(corpus), "--dst", str(second), "--seed", "0"])
        assert (first / "split_manifest.json").read_text() == (
            second / "split_manifest.json"
        ).read_text()

    def test_data_yaml_preserves_class_order(self, corpus: Path, tmp_path: Path) -> None:
        """Class indices are baked into every label file — order is load-bearing."""
        dst = tmp_path / "out"
        bd.main(["--src", str(corpus), "--dst", str(dst), "--seed", "0"])
        text = (dst / "data.yaml").read_text(encoding="utf-8")
        labels = (
            Path(__file__).resolve().parents[2]
            / "src/urolens_ai/models/yolov8/labels.txt"
        ).read_text(encoding="utf-8").split()
        assert [line[2:] for line in text.splitlines() if line.startswith("- ")] == labels
