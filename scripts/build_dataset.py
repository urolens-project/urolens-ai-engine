"""
scripts/build_dataset.py
------------------------
Rebuild the UroLens detection dataset as a leak-free, deduplicated corpus.

The Roboflow export at UroLens-3 is a merge of at least six acquired sources and
carries pre-applied augmentation -- brightness/contrast variants AND flips and
rotations -- which Roboflow then split at image level. The result is that roughly
half of all images have a near-duplicate sibling in a different split, so every
accuracy number measured against it is inflated.

This script rebuilds the corpus:

    1. Fingerprint every image (orientation-aware, brightness/contrast invariant).
    2. Union same-source and verified visual duplicates into groups.
    3. Collapse each visual group to a single representative image.
    4. Split by group -- never by image -- stratified by source family and class.
    5. Assert that no group, source name, or visual duplicate spans two splits.

Usage:
    python scripts/build_dataset.py --src <UroLens-3> --dst <UroLens-4-clean>
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Tuning constants -- validated against UroLens-3, see module docstring
# ---------------------------------------------------------------------------

SPLITS = ("train", "valid", "test")

DESC_SIZE = 32          # coarse descriptor edge, in pixels
HP_SIZE = 128           # high-pass verification edge, in pixels
HP_SIGMA = 6.0          # gaussian sigma removed to strip the illumination gradient
COARSE_MIN_CORR = 0.97  # coarse correlation required to become a candidate pair
HP_MIN_CORR = 0.50      # particle-detail correlation required to confirm a duplicate
FLIP_MARGIN = 0.15      # a mirrored match must beat the upright match by this much
LABEL_MATCH_MIN = 0.80  # fraction of boxes that must land on transformed coordinates
LABEL_TOL = 0.03        # coordinate tolerance when comparing transformed boxes
TEXTURE_PCT = 25        # images below this texture percentile are too blank to match
MAX_BUCKET = 50         # hash buckets larger than this are degenerate (blank fields)

ORIENTATIONS = ("identity", "fliplr", "flipud", "rot180")

CLASS_NAMES = (
    "bacteria",
    "crystals",
    "epithelial-cells",
    "erythrocytes",
    "leukocytes",
    "mucus-threads",
    "sperm-cells",
    "trichomonas-vaginalis",
    "urinary-casts",
    "yeast",
)


# ---------------------------------------------------------------------------
# Source families
# ---------------------------------------------------------------------------

# (family name, filename pattern, whether a specimen id is recoverable)
_FAMILY_PATTERNS: tuple[tuple[str, "re.Pattern[str]", bool], ...] = (
    ("field_bmp", re.compile(r"^\d+_\d{6}_\d{2}_\d{2}_\d{2}_(?:-|QC)"), True),
    ("win_camera", re.compile(r"^WIN_\d{8}_\d{2}_\d{2}_\d{2}"), True),
    ("timestamp_field", re.compile(r"^\d{14}_\d+$"), True),
    ("nh_nl_id", re.compile(r"^n[hl]\d+$"), False),
    ("pure_number", re.compile(r"^\d+$"), False),
)


def family_of(basename: str) -> tuple[str, bool]:
    """Return (family name, whether a specimen id is recoverable from the name)."""
    for name, pattern, groupable in _FAMILY_PATTERNS:
        if pattern.match(basename):
            return name, groupable
    return "other", False


def specimen_key(basename: str, groupable: bool) -> str | None:
    """
    Patient/session key for the families that encode one.

    Those filenames end in a field number: several microscope fields are captured
    per specimen and must never be split apart. Returns None for sources that
    encode no provenance at all -- 75% of this corpus, which is a known and
    documented limitation rather than a solved problem.
    """
    if not groupable:
        return None
    head, _, _tail = basename.rpartition("_")
    return head if head else basename


def source_basename(filename: str) -> str:
    """Strip Roboflow's '_jpg.rf.<hash>.jpg' suffix to recover the source name."""
    return filename.split("_jpg.rf.")[0]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

Box = tuple[int, float, float, float, float]


@dataclass
class ImageRec:
    index: int
    split: str
    filename: str
    image_path: Path
    label_path: Path
    basename: str = ""
    family: str = ""
    groupable: bool = False
    boxes: list[Box] = field(default_factory=list)
    texture: float = 0.0
    readable: bool = True

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    @property
    def dominant_class(self) -> str:
        if not self.boxes:
            return "empty"
        counts = collections.Counter(b[0] for b in self.boxes)
        return CLASS_NAMES[counts.most_common(1)[0][0]]


def read_labels(path: Path) -> list[Box]:
    """Parse a YOLO label file. Missing or empty files yield no boxes."""
    if not path.exists():
        return []
    boxes: list[Box] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                continue
            boxes.append((cls, x, y, w, h))
    return boxes


def collect(src: Path) -> list[ImageRec]:
    """Enumerate every image in the source export, with its labels attached."""
    recs: list[ImageRec] = []
    for split in SPLITS:
        image_dir = src / split / "images"
        label_dir = src / split / "labels"
        if not image_dir.is_dir():
            continue
        for filename in sorted(os.listdir(image_dir)):
            stem = filename.rsplit(".", 1)[0]
            rec = ImageRec(
                index=len(recs),
                split=split,
                filename=filename,
                image_path=image_dir / filename,
                label_path=label_dir / f"{stem}.txt",
            )
            rec.basename = source_basename(filename)
            rec.family, rec.groupable = family_of(rec.basename)
            rec.boxes = read_labels(rec.label_path)
            recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def _load_gray(path: Path, size: int) -> np.ndarray:
    image = Image.open(path)
    # draft() lets libjpeg decode at a reduced DCT scale -- much cheaper than a
    # full decode followed by a resize.
    image.draft("L", (size * 2, size * 2))
    return np.asarray(
        image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32
    )


def _znorm(array: np.ndarray) -> np.ndarray:
    std = float(array.std())
    return (array - array.mean()) / (std if std > 1e-6 else 1.0)


def orient(array: np.ndarray, transform: int) -> np.ndarray:
    """Apply one of the four dihedral transforms this dataset actually contains."""
    if transform == 0:
        return array
    if transform == 1:
        return array[:, ::-1]
    if transform == 2:
        return array[::-1, :]
    return array[::-1, ::-1]


def _hash64(descriptor: np.ndarray) -> int:
    """Median-threshold hash: balanced bits, invariant to brightness and contrast."""
    block = DESC_SIZE // 8
    pooled = descriptor.reshape(8, block, 8, block).mean(axis=(1, 3))
    bits = (pooled > np.median(pooled)).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def fingerprint(recs: list[ImageRec], verbose: bool = True) -> np.ndarray:
    """
    Compute a coarse descriptor per image and populate rec.texture.

    Returns an (N, DESC_SIZE**2) float32 array of z-normalised descriptors. The
    z-normalisation is what makes this invariant to the brightness and contrast
    variants baked into the export.
    """
    descriptors = np.zeros((len(recs), DESC_SIZE * DESC_SIZE), dtype=np.float32)
    for i, rec in enumerate(recs):
        try:
            raw = _load_gray(rec.image_path, DESC_SIZE)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            rec.readable = False
            print(f"  WARNING: could not read {rec.filename}: {exc}", file=sys.stderr)
            continue
        rec.texture = float(raw.std())
        descriptors[i] = _znorm(raw).ravel()
        if verbose and i and i % 5000 == 0:
            print(f"  fingerprinted {i}/{len(recs)}", flush=True)
    return descriptors


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class HighPassCache:
    """
    Lazily computes and caches high-pass descriptors.

    The raw images are very low contrast (grayscale sigma around 4.4 of 255), so a
    plain descriptor correlates on the microscope's illumination gradient rather
    than on the specimen. Subtracting a heavy gaussian isolates particle detail,
    which is what actually separates a true duplicate from two similar empty fields.
    """

    def __init__(self, recs: list[ImageRec], capacity: int = 4000) -> None:
        self._recs = recs
        self._capacity = capacity
        self._cache: "collections.OrderedDict[int, np.ndarray]" = collections.OrderedDict()

    def get(self, index: int) -> np.ndarray:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        raw = _load_gray(self._recs[index].image_path, HP_SIZE)
        detail = raw - gaussian_filter(raw, sigma=HP_SIGMA)
        std = float(detail.std())
        descriptor = ((detail - detail.mean()) / (std if std > 1e-6 else 1.0)).astype(
            np.float16
        )
        self._cache[index] = descriptor
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return descriptor

    def correlation(self, i: int, j: int, transform: int) -> float:
        a = self.get(i).astype(np.float32)
        b = orient(self.get(j).astype(np.float32), transform)
        return float((a * b).mean())


def transform_box(box: Box, transform: int) -> Box:
    """Mirror a YOLO box under one of the four dihedral transforms."""
    cls, x, y, w, h = box
    if transform in (1, 3):
        x = 1.0 - x
    if transform in (2, 3):
        y = 1.0 - y
    return (cls, x, y, w, h)


def label_agreement(a: ImageRec, b: ImageRec, transform: int) -> float | None:
    """
    Fraction of b's boxes that land on a's boxes under `transform`.

    Returns None when the comparison is not meaningful -- either side empty, or the
    box counts differ. This is the strongest available evidence of a flip: box
    geometry cannot agree under mirroring by coincidence.
    """
    if not a.boxes or not b.boxes or len(a.boxes) != len(b.boxes):
        return None
    left = sorted(a.boxes)
    right = sorted(transform_box(box, transform) for box in b.boxes)
    hits = sum(
        1
        for p, q in zip(left, right)
        if p[0] == q[0]
        and abs(p[1] - q[1]) <= LABEL_TOL
        and abs(p[2] - q[2]) <= LABEL_TOL
    )
    return hits / len(left)


@dataclass
class DuplicatePair:
    i: int
    j: int
    transform: int
    coarse: float
    upright: float
    highpass: float
    label_match: float | None
    reason: str


def find_visual_duplicates(
    recs: list[ImageRec], descriptors: np.ndarray, verbose: bool = True
) -> list[DuplicatePair]:
    """
    Find visually duplicated image pairs that do NOT share a source filename.

    Same-name copies need no verification -- Roboflow derived them from one source
    file, so they are unioned directly by the caller. What this has to catch is the
    harder case: the same field stored under a different name, possibly mirrored.

    Two stages: bucket on all four orientation hashes so a mirrored pair collides,
    screen with the coarse descriptor, then confirm with the high-pass test and,
    where box counts allow, mirrored label geometry.
    """
    buckets: dict[int, list[int]] = collections.defaultdict(list)
    for i, rec in enumerate(recs):
        if not rec.readable:
            continue
        descriptor = descriptors[i].reshape(DESC_SIZE, DESC_SIZE)
        for transform in range(4):
            buckets[_hash64(orient(descriptor, transform))].append(i)

    textures = [r.texture for r in recs if r.texture > 0]
    texture_floor = float(np.percentile(textures, TEXTURE_PCT)) if textures else 0.0
    if verbose:
        print(f"  texture floor (p{TEXTURE_PCT}) = {texture_floor:.2f}", flush=True)

    candidates: set[tuple[int, int]] = set()
    degenerate = 0
    for members in buckets.values():
        if len(members) > MAX_BUCKET:
            degenerate += 1
            continue
        unique = sorted(set(members))
        for a_pos in range(len(unique)):
            for b_pos in range(a_pos + 1, len(unique)):
                candidates.add((unique[a_pos], unique[b_pos]))
    if verbose:
        print(
            f"  {len(candidates)} candidate pairs "
            f"({degenerate} degenerate buckets skipped)",
            flush=True,
        )

    cache = HighPassCache(recs)
    pairs: list[DuplicatePair] = []
    for done, (i, j) in enumerate(sorted(candidates)):
        if recs[i].basename == recs[j].basename:
            continue  # already unioned by source name
        if recs[i].texture < texture_floor or recs[j].texture < texture_floor:
            continue  # too blank for content matching to mean anything

        scores = [
            float(
                np.dot(
                    descriptors[i],
                    orient(descriptors[j].reshape(DESC_SIZE, DESC_SIZE), t).ravel(),
                )
                / (DESC_SIZE * DESC_SIZE)
            )
            for t in range(4)
        ]
        best = int(np.argmax(scores))
        if scores[best] < COARSE_MIN_CORR:
            continue
        if best != 0 and scores[best] - scores[0] < FLIP_MARGIN:
            continue

        highpass = cache.correlation(i, j, best)
        if highpass < HP_MIN_CORR:
            continue
        match = label_agreement(recs[i], recs[j], best)
        if match is not None and match < LABEL_MATCH_MIN:
            continue

        pairs.append(
            DuplicatePair(
                i, j, best, scores[best], scores[0], highpass, match,
                "verified_visual_duplicate",
            )
        )
        if verbose and done and done % 20000 == 0:
            print(f"  screened {done}/{len(candidates)} candidates", flush=True)
    return pairs


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def groups(self, size: int) -> dict[int, list[int]]:
        out: dict[int, list[int]] = collections.defaultdict(list)
        for i in range(size):
            out[self.find(i)].append(i)
        return out


def build_groups(
    recs: list[ImageRec], pairs: list[DuplicatePair]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Union images at two levels, because they answer different questions.

    visual group -- these are the same picture, so keep only one of them.
    split group  -- these may be the same patient, so keep them in one split.

    Split groups are a superset: every visual group plus specimen provenance
    wherever the filename encodes it.
    """
    visual = UnionFind(len(recs))
    by_name: dict[str, list[int]] = collections.defaultdict(list)
    for i, rec in enumerate(recs):
        by_name[rec.basename].append(i)
    for members in by_name.values():
        for other in members[1:]:
            visual.union(members[0], other)
    for pair in pairs:
        visual.union(pair.i, pair.j)

    split_uf = UnionFind(len(recs))
    for i in range(len(recs)):
        split_uf.union(i, visual.find(i))
    by_specimen: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, rec in enumerate(recs):
        key = specimen_key(rec.basename, rec.groupable)
        if key is not None:
            by_specimen[(rec.family, key)].append(i)
    for members in by_specimen.values():
        for other in members[1:]:
            split_uf.union(members[0], other)

    return visual.groups(len(recs)), split_uf.groups(len(recs))


def choose_representative(recs: list[ImageRec], members: list[int]) -> int:
    """
    Pick one image to stand for a visual group.

    Most boxes wins: where copies disagree, a blank copy of an image annotated
    elsewhere is a missed annotation, not a genuinely empty field. Ties break on
    filename so the choice is reproducible across runs.
    """
    return max(members, key=lambda i: (recs[i].n_boxes, recs[i].filename))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def stratified_split(
    recs: list[ImageRec],
    split_groups: dict[int, list[int]],
    representatives: dict[int, int],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[int, str]:
    """
    Assign whole groups to splits, stratified by source family and dominant class.

    Stratifying on family keeps every acquired source represented in validation and
    test; stratifying on class keeps rare particles (trichomonas, mucus-threads)
    from vanishing out of the evaluation sets entirely.
    """
    strata: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for root, members in split_groups.items():
        kept = [representatives[m] for m in members if m in representatives]
        if not kept:
            continue
        # Characterise the group by its richest representative.
        lead = max(kept, key=lambda i: recs[i].n_boxes)
        strata[(recs[lead].family, recs[lead].dominant_class)].append(root)

    rng = random.Random(seed)
    assignment: dict[int, str] = {}
    train_r, valid_r, _test_r = ratios
    for key in sorted(strata):
        roots = sorted(strata[key])
        rng.shuffle(roots)
        n = len(roots)
        n_train = int(round(n * train_r))
        n_valid = int(round(n * valid_r))
        # Guarantee at least one group in valid and test once a stratum can afford it.
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_valid = max(1, min(n_valid, n - n_train - 1))
        for pos, root in enumerate(roots):
            if pos < n_train:
                assignment[root] = "train"
            elif pos < n_train + n_valid:
                assignment[root] = "valid"
            else:
                assignment[root] = "test"
    return assignment


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def place_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass  # cross-volume or unsupported -- fall back to a copy
    shutil.copy2(source, target)


def write_dataset(
    recs: list[ImageRec],
    kept: dict[int, str],
    dst: Path,
    mode: str,
) -> None:
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
    for index, split in kept.items():
        rec = recs[index]
        stem = rec.filename.rsplit(".", 1)[0]
        place_file(rec.image_path, dst / split / "images" / rec.filename, mode)
        label_target = dst / split / "labels" / f"{stem}.txt"
        if rec.label_path.exists():
            place_file(rec.label_path, label_target, mode)
        else:
            label_target.write_text("", encoding="utf-8")

    data_yaml = [
        "# Generated by scripts/build_dataset.py -- do not edit by hand.",
        "# Class order is load-bearing: indices are baked into every label file.",
        "names:",
        *[f"- {name}" for name in CLASS_NAMES],
        f"nc: {len(CLASS_NAMES)}",
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
    ]
    (dst / "data.yaml").write_text("\n".join(data_yaml), encoding="utf-8")


def verify_disjoint(
    recs: list[ImageRec], kept: dict[int, str], pairs: list[DuplicatePair]
) -> list[str]:
    """
    Fail loudly if the leak could reappear.

    This is the guard that makes the whole rebuild worth doing -- without it the
    next regeneration could silently reintroduce exactly the bug being fixed.
    """
    errors: list[str] = []

    by_name: dict[str, set[str]] = collections.defaultdict(set)
    for index, split in kept.items():
        by_name[recs[index].basename].add(split)
    straddling = {name for name, splits in by_name.items() if len(splits) > 1}
    if straddling:
        errors.append(
            f"{len(straddling)} source names span >1 split, e.g. "
            f"{sorted(straddling)[:3]}"
        )

    by_specimen: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for index, split in kept.items():
        rec = recs[index]
        key = specimen_key(rec.basename, rec.groupable)
        if key is not None:
            by_specimen[(rec.family, key)].add(split)
    bad_specimens = {k for k, splits in by_specimen.items() if len(splits) > 1}
    if bad_specimens:
        errors.append(
            f"{len(bad_specimens)} specimens span >1 split, e.g. "
            f"{sorted(bad_specimens)[:3]}"
        )

    for pair in pairs:
        left, right = kept.get(pair.i), kept.get(pair.j)
        if left and right and left != right:
            errors.append(
                f"visual duplicate kept in two splits: "
                f"{recs[pair.i].filename} ({left}) / {recs[pair.j].filename} ({right})"
            )
            break
    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Roboflow export root")
    parser.add_argument("--dst", type=Path, required=True, help="output dataset root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.70, 0.15, 0.15),
        metavar=("TRAIN", "VALID", "TEST"),
    )
    parser.add_argument(
        "--link",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="how to materialise images (hardlink costs no extra disk)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="analyse and report, but write no dataset",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dst: Path = args.dst

    print(f"Reading {args.src}", flush=True)
    recs = collect(args.src)
    if not recs:
        print("ERROR: no images found", file=sys.stderr)
        return 2
    total_boxes = sum(r.n_boxes for r in recs)
    print(f"  {len(recs)} images, {total_boxes} boxes", flush=True)

    print("Fingerprinting", flush=True)
    descriptors = fingerprint(recs)

    print("Detecting cross-name and mirrored duplicates", flush=True)
    pairs = find_visual_duplicates(recs, descriptors)
    flips = [p for p in pairs if p.transform != 0]
    print(
        f"  {len(pairs)} verified visual duplicate pairs "
        f"({len(flips)} mirrored)",
        flush=True,
    )

    visual_groups, split_groups = build_groups(recs, pairs)
    representatives: dict[int, int] = {}
    conflicts: list[list[object]] = []
    for root, members in visual_groups.items():
        rep = choose_representative(recs, members)
        for m in members:
            representatives[m] = rep
        counts = {recs[m].n_boxes for m in members}
        if len(counts) > 1:
            conflicts.append(
                [
                    recs[rep].basename,
                    len(members),
                    ";".join(str(recs[m].n_boxes) for m in sorted(members)),
                    ";".join(recs[m].split for m in sorted(members)),
                    recs[rep].filename,
                    recs[rep].n_boxes,
                    1 if min(counts) == 0 and max(counts) > 0 else 0,
                ]
            )
    unique_reps = sorted(set(representatives.values()))
    print(
        f"  {len(recs)} images -> {len(unique_reps)} unique visual sources "
        f"({len(split_groups)} split-atomic groups)",
        flush=True,
    )
    print(f"  {len(conflicts)} groups disagree on box count", flush=True)

    assignment = stratified_split(
        recs, split_groups, representatives, tuple(args.ratios), args.seed
    )
    kept: dict[int, str] = {}
    for root, members in split_groups.items():
        split = assignment.get(root)
        if split is None:
            continue
        for rep in {representatives[m] for m in members if m in representatives}:
            kept[rep] = split

    counts = collections.Counter(kept.values())
    print(
        "  split: "
        + ", ".join(f"{s}={counts.get(s, 0)}" for s in SPLITS)
        + f" (total {len(kept)})",
        flush=True,
    )

    errors = verify_disjoint(recs, kept, pairs)
    if errors:
        print("DISJOINTNESS CHECK FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("  disjointness check passed", flush=True)

    if args.dry_run:
        print("Dry run -- nothing written.", flush=True)
        return 0

    print(f"Writing {dst}", flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    write_dataset(recs, kept, dst, args.link)

    manifest = {
        "seed": args.seed,
        "ratios": list(args.ratios),
        "source": str(args.src),
        "source_images": len(recs),
        "source_boxes": total_boxes,
        "unique_visual_sources": len(unique_reps),
        "kept_images": len(kept),
        "kept_boxes": sum(recs[i].n_boxes for i in kept),
        "verified_duplicate_pairs": len(pairs),
        "mirrored_pairs": len(flips),
        "box_count_conflicts": len(conflicts),
        "splits": {s: counts.get(s, 0) for s in SPLITS},
        "assignments": {
            recs[i].filename: {
                "split": split,
                "family": recs[i].family,
                "specimen": specimen_key(recs[i].basename, recs[i].groupable),
                "boxes": recs[i].n_boxes,
            }
            for i, split in sorted(kept.items())
        },
    }
    (dst / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    with (dst / "duplicates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_a", "split_a", "file_b", "split_b", "transform",
                "coarse_corr", "upright_corr", "highpass_corr", "label_match",
            ]
        )
        for p in pairs:
            writer.writerow(
                [
                    recs[p.i].filename, recs[p.i].split,
                    recs[p.j].filename, recs[p.j].split,
                    ORIENTATIONS[p.transform],
                    f"{p.coarse:.4f}", f"{p.upright:.4f}", f"{p.highpass:.4f}",
                    "" if p.label_match is None else f"{p.label_match:.2f}",
                ]
            )

    with (dst / "conflicts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_name", "n_copies", "box_counts", "splits",
                "kept_file", "kept_boxes", "blank_vs_annotated",
            ]
        )
        writer.writerows(sorted(conflicts, key=lambda r: str(r[0])))

    print(
        f"Done. {len(kept)} images, {manifest['kept_boxes']} boxes.\n"
        f"  {dst / 'split_manifest.json'}\n"
        f"  {dst / 'duplicates.csv'}\n"
        f"  {dst / 'conflicts.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
