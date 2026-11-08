"""
scripts/class_balance.py
------------------------
Class distribution across splits, on both axes that matter for detection:

  * INSTANCE count -- how many boxes of that class exist. Drives the loss.
  * IMAGE count    -- how many images contain the class at all. Drives how
                      often the model encounters it in context.

These come apart badly on this dataset: bacteria has plenty of instances but
they are packed into very few images, so the detector rarely meets one. Fixing
instance imbalance (loss weighting) and image imbalance (oversampling) are
different interventions.

    python scripts/class_balance.py --root C:/Users/Harley/UroLens/UroLens-5-hires
    python scripts/class_balance.py --root ... --list train_oversampled.txt
"""

from __future__ import annotations

import argparse
import collections
import glob
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")


def scan(label_paths: list[Path]) -> tuple[collections.Counter, collections.Counter]:
    inst: collections.Counter[int] = collections.Counter()
    imgs: collections.Counter[int] = collections.Counter()
    for lp in label_paths:
        if not lp.exists():
            continue
        seen = set()
        for line in lp.read_text().splitlines():
            parts = line.split()
            if parts:
                c = int(parts[0])
                inst[c] += 1
                seen.add(c)
        for c in seen:
            imgs[c] += 1
    return inst, imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--list", default=None, help="a train list .txt to score instead of train/")
    args = ap.parse_args()

    names = yaml.safe_load((args.root / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    if args.list:
        entries = (args.root / args.list).read_text().split()
        labels = [
            Path(str(Path(e).parent).replace("images", "labels")) / (Path(e).stem + ".txt")
            for e in entries
        ]
        sections = {f"LIST {args.list}": labels}
    else:
        sections = {
            s: [Path(p) for p in glob.glob(str(args.root / s / "labels" / "*.txt"))]
            for s in SPLITS
        }

    for title, labels in sections.items():
        inst, imgs = scan(labels)
        if not inst:
            continue
        ti, tm = sum(inst.values()), len(labels)
        print(f"\n=== {title}  ({tm} image entries, {ti} instances) ===")
        print(f"{'class':<24}{'inst':>8}{'inst%':>8}{'images':>8}{'img%':>7}{'per img':>9}")
        for c, n in inst.most_common():
            im = imgs[c]
            print(
                f"{names[c]:<24}{n:>8}{100*n/ti:>7.1f}%{im:>8}{100*im/tm:>6.1f}%"
                f"{n/im if im else 0:>9.1f}"
            )
        hi_i, lo_i = max(inst.values()), min(inst.values())
        hi_m = max(imgs[c] for c in inst)
        lo_m = min(imgs[c] for c in inst)
        print(f"{'':<24}{'':>8}{'':>8}")
        print(f"  instance imbalance : {hi_i/lo_i:.1f} : 1")
        print(f"  image    imbalance : {hi_m/lo_m:.1f} : 1")


if __name__ == "__main__":
    main()
