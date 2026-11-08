"""
scripts/build_oversampled_list.py
---------------------------------
Emit a YOLO train list that repeats images containing rare classes.

bacteria (2.5% of train images) and mucus-threads (3.7%) are severely
under-represented at the IMAGE level even though bacteria has ~6k instances --
they are concentrated in 262 and 386 images respectively. The detector sees
them so rarely it under-fires (bacteria: precision 0.890, recall 0.404).

Repeating those images in the train list raises their effective sampling rate
without touching the images on disk. Ultralytics accepts a .txt of image paths
in place of a directory for `train:`.

Usage:
    python scripts/build_oversampled_list.py \
        --root C:/Users/Harley/UroLens/UroLens-4-clean
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
from pathlib import Path

import yaml

# class index -> how many times each containing image appears in the list
REPEATS: dict[int, int] = {0: 5, 5: 4}  # bacteria, mucus-threads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    root: Path = args.root
    split_dir = root / args.split
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]

    images = sorted(glob.glob(str(split_dir / "images" / "*.*")))
    lines: list[str] = []
    boosted: collections.Counter[int] = collections.Counter()

    for image_path in images:
        stem = Path(image_path).stem
        label_path = split_dir / "labels" / f"{stem}.txt"
        absolute = Path(image_path).resolve().as_posix()
        lines.append(absolute)

        if not label_path.exists():
            continue
        classes = {
            int(line.split()[0])
            for line in label_path.read_text().splitlines()
            if line.split()
        }
        extra = max([REPEATS.get(c, 1) for c in classes] + [1]) - 1
        if extra:
            for c in classes & REPEATS.keys():
                boosted[c] += 1
            lines.extend([absolute] * extra)

    list_path = root / f"{args.split}_oversampled.txt"
    list_path.write_text("\n".join(lines) + "\n")

    data = yaml.safe_load((root / "data.yaml").read_text())
    data["train"] = list_path.resolve().as_posix()
    data["val"] = (root / "valid" / "images").resolve().as_posix()
    data["test"] = (root / "test" / "images").resolve().as_posix()
    yaml_path = root / "data_oversampled.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))

    print(f"base images      : {len(images)}")
    for c, n in sorted(boosted.items()):
        print(f"  {names[c]:<16} {n} images x{REPEATS[c]}")
    print(f"total entries    : {len(lines)} (+{100 * (len(lines) / len(images) - 1):.0f}%)")
    print(f"wrote            : {list_path}")
    print(f"wrote            : {yaml_path}")


if __name__ == "__main__":
    main()
