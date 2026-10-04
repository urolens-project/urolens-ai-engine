"""
scripts/rebuild_native_res.py
-----------------------------
Rebuild the clean dataset at native source resolution.

The v3 Roboflow export applied `resize: Fit within 640x640`, so every local
image is <=640px. v4 was generated without that step. 55.7% of the sources are
larger -- up to 3088x2064 -- and 82% of bacteria / 97% of mucus-threads images
sit in the high-resolution family, which is exactly where detection is weakest.

This does NOT re-run deduplication. It reuses two things already established:

  * the split assignment in `split_manifest.json` (specimen-disjoint, verified)
  * the conflict-resolved LABELS in UroLens-4-clean (936 box-count conflicts
    were adjudicated to produce them; the v4 export's labels are raw)

so only the image PIXELS come from v4. Labels are normalised YOLO coordinates
and transfer across resolutions unchanged.

Matching is on the filename stem before Roboflow's `.rf.<hash>` suffix, which
derives from the original upload name and is stable across versions.

Usage:
    python scripts/rebuild_native_res.py \
        --clean  C:/Users/Harley/UroLens/UroLens-4-clean \
        --native C:/Users/Harley/UroLens/UroLens-4-native \
        --out    C:/Users/Harley/UroLens/UroLens-5-hires
"""

from __future__ import annotations

import argparse
import collections
import glob
import shutil
from pathlib import Path

import yaml

try:
    from PIL import Image
except ImportError:
    Image = None

SPLITS = ("train", "valid", "test")


def stem_key(name: str) -> str:
    """Filename stem before Roboflow's `.rf.<hash>` suffix."""
    base = Path(name).name
    return base.split(".rf.")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # index every image in the v4 export by stem, regardless of which split
    # Roboflow put it in -- our split comes from the manifest, not from them.
    native_index: dict[str, Path] = {}
    for path in glob.glob(str(args.native / "**" / "images" / "*.*"), recursive=True):
        native_index.setdefault(stem_key(path), Path(path))
    print(f"native export indexed : {len(native_index)} images")

    stats: collections.Counter[str] = collections.Counter()
    upgraded_px: list[tuple[int, int]] = []

    for split in SPLITS:
        out_img = args.out / split / "images"
        out_lbl = args.out / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img in sorted(glob.glob(str(args.clean / split / "images" / "*.*"))):
            src = Path(img)
            label = args.clean / split / "labels" / f"{src.stem}.txt"
            native = native_index.get(stem_key(src.name))

            if native is None:
                shutil.copy2(src, out_img / src.name)   # fall back to the 640 copy
                stats[f"{split}:fallback"] += 1
            else:
                shutil.copy2(native, out_img / (src.stem + native.suffix))
                stats[f"{split}:native"] += 1
                if Image is not None and stats[f"{split}:native"] % 500 == 0:
                    upgraded_px.append(Image.open(native).size)

            if label.exists():
                shutil.copy2(label, out_lbl / f"{src.stem}.txt")
                stats[f"{split}:labels"] += 1

    data = yaml.safe_load((args.clean / "data.yaml").read_text())
    data["train"] = (args.out / "train" / "images").resolve().as_posix()
    data["val"] = (args.out / "valid" / "images").resolve().as_posix()
    data["test"] = (args.out / "test" / "images").resolve().as_posix()
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    shutil.copy2(args.clean / "split_manifest.json", args.out / "split_manifest.json")

    print()
    for k in sorted(stats):
        print(f"  {k:<20} {stats[k]}")
    if upgraded_px:
        print(f"\n  sampled native sizes: {upgraded_px[:8]}")
    print(f"\nwrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
