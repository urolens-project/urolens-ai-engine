"""
scripts/build_gate_dataset.py
-----------------------------
Build the image-classification dataset for the input gate ("is this a urine
microscopy image at all?").

Why this exists: the detector only knows the ten particle classes, so on a photo
that is not a slide it still has to label *something*. A selfie in a classroom
came back as "2 sperm cells" -- plain bright wall patches look like an empty
microscope field. The gate rejects such images before the detector ever runs.

Classes (Ultralytics classify layout, one folder per class):

    slide       UroLens-4-clean images, kept in their original split so the
                gate's test set is exactly the detector's test set, plus
                OpenUrine images (a different microscope) split 70/15/15 so the
                gate learns that other slide setups still count as slides.
    not_slide   COCO val2017 images (people, rooms, objects, screens) plus any
                extra photos passed with --extra-neg, split 70/15/15.

All of UroLens test is kept: the acceptance bar is that >=99.9% of real test
slides pass. Train positives are subsampled to keep the classes balanced.

Usage:
    python scripts/build_gate_dataset.py \
        --urolens <UroLens-4-clean> --openurine <dir> --coco <val2017 dir> \
        --dst <gate-cls> [--extra-neg <dir>]
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def split_70_15_15(items: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    items = items[:]
    rng.shuffle(items)
    n_train = int(len(items) * 0.70)
    n_val = int(len(items) * 0.15)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def place(files: list[Path], dst: Path, prefix: str) -> None:
    """Hard-link (or copy, across volumes) files into dst with a source prefix."""
    dst.mkdir(parents=True, exist_ok=True)
    for src in files:
        target = dst / f"{prefix}_{src.name}"
        if target.exists():
            continue
        try:
            target.hardlink_to(src)
        except OSError:
            shutil.copy2(src, target)


def build(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    rng = random.Random(args.seed)
    plan: dict[str, dict[str, list[tuple[str, list[Path]]]]] = {
        s: {"slide": [], "not_slide": []} for s in SPLITS
    }

    # Positives: UroLens keeps its own leak-free split (valid -> val).
    uro = {
        "train": list_images(args.urolens / "train" / "images"),
        "val": list_images(args.urolens / "valid" / "images"),
        "test": list_images(args.urolens / "test" / "images"),
    }
    uro["train"] = rng.sample(uro["train"], min(args.max_train_pos, len(uro["train"])))
    uro["val"] = rng.sample(uro["val"], min(args.max_val_pos, len(uro["val"])))
    for s in SPLITS:
        plan[s]["slide"].append(("uro", uro[s]))

    for s, files in split_70_15_15(list_images(args.openurine), rng).items():
        plan[s]["slide"].append(("openurine", files))

    # Negatives.
    coco = list_images(args.coco)
    coco = rng.sample(coco, min(args.max_coco, len(coco)))
    for s, files in split_70_15_15(coco, rng).items():
        plan[s]["not_slide"].append(("coco", files))
    if args.extra_neg:
        for s, files in split_70_15_15(list_images(args.extra_neg), rng).items():
            plan[s]["not_slide"].append(("own", files))

    counts: dict[str, dict[str, int]] = {}
    for s in SPLITS:
        for cls, groups in plan[s].items():
            for prefix, files in groups:
                place(files, args.dst / s / cls, prefix)
                counts.setdefault(f"{s}/{cls}", {})[prefix] = len(files)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urolens", type=Path, required=True, help="UroLens-4-clean root")
    parser.add_argument("--openurine", type=Path, required=True, help="folder of OpenUrine images")
    parser.add_argument("--coco", type=Path, required=True, help="COCO val2017 image folder")
    parser.add_argument("--extra-neg", type=Path, default=None, help="own non-slide photos")
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--max-train-pos", type=int, default=3000)
    parser.add_argument("--max-val-pos", type=int, default=600)
    parser.add_argument("--max-coco", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dst.exists() and any(args.dst.iterdir()):
        print(f"refusing to write into non-empty {args.dst}", file=sys.stderr)
        return 1
    for key, by_source in sorted(build(args).items()):
        print(f"{key:18s} {sum(by_source.values()):5d}  {by_source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
