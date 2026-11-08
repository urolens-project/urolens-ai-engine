"""
scripts/build_negatives_dataset.py
----------------------------------
Build the v5 detector dataset: UroLens-4-clean plus non-slide negatives.

The detector has never seen a non-slide image, which is how a selfie came back
as "2 sperm cells". YOLO treats an image with an empty label file as pure
background, so adding negatives teaches it to stay silent on them.

Negatives are drawn ONLY from the input gate's train split
(gate-cls/train/not_slide). The gate's val/test negatives stay unseen by both
models, so one held-out set measures the gate and the detector's false
positives alike.

Nothing in UroLens-4-clean is copied or modified: the train split is a .txt list
of the original images plus the negatives, and val/test point at the originals.
That keeps v5 directly comparable to v2-clean -- negatives are the only change.

Usage:
    python scripts/build_negatives_dataset.py \
        --base C:/Users/Harley/UroLens/UroLens-4-clean \
        --negatives C:/Users/Harley/UroLens/gate-cls/train/not_slide \
        --dst C:/Users/Harley/UroLens/UroLens-5-neg --count 1000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--negatives", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base: Path = args.base.resolve()
    dst: Path = args.dst.resolve()
    pool = list_images(args.negatives)
    if args.count > len(pool):
        raise SystemExit(f"only {len(pool)} negatives available, asked for {args.count}")
    chosen = sorted(random.Random(args.seed).sample(pool, args.count))

    # Ultralytics finds labels by swapping /images/ for /labels/ in the path.
    neg_images = dst / "negatives" / "images"
    neg_labels = dst / "negatives" / "labels"
    neg_images.mkdir(parents=True, exist_ok=True)
    neg_labels.mkdir(parents=True, exist_ok=True)
    for src in chosen:
        shutil.copy2(src, neg_images / src.name)
        (neg_labels / f"{src.stem}.txt").write_text("")

    base_train = list_images(base / "train" / "images")
    lines = [p.as_posix() for p in base_train]
    lines += [(neg_images / p.name).as_posix() for p in chosen]
    train_list = dst / "train.txt"
    train_list.write_text("\n".join(lines) + "\n")

    data = yaml.safe_load((base / "data.yaml").read_text())
    data["train"] = train_list.as_posix()
    data["val"] = (base / "valid" / "images").as_posix()
    data["test"] = (base / "test" / "images").as_posix()
    (dst / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    # train.py's guard_dataset() looks for this; it also records what went in.
    manifest = json.loads((base / "split_manifest.json").read_text(encoding="utf-8"))
    (dst / "split_manifest.json").write_text(
        json.dumps(
            {
                "kept_images": manifest["kept_images"] + len(chosen),
                "source_images": manifest["source_images"],
                "seed": manifest["seed"],
                "base": base.as_posix(),
                "negatives_source": args.negatives.resolve().as_posix(),
                "negatives_seed": args.seed,
                "negatives": [p.name for p in chosen],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"train images     : {len(base_train)} + {len(chosen)} negatives")
    print(f"wrote            : {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
