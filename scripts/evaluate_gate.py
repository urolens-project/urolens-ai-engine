"""
scripts/evaluate_gate.py
------------------------
Measure the input gate through the same path production uses
(preprocessing.normalise() -> InputGate) and pick its threshold.

The acceptance bar is on real slides, not on accuracy: the gate must pass
>=99.9% of UroLens test slides. The threshold reported is the highest slide-
probability cut-off that still meets that bar; the script then reports what it
does to every other source (OpenUrine = a different microscope, COCO = the
non-slide images the gate must reject).

Usage:
    python scripts/evaluate_gate.py --weights runs/classify/gate-v1/weights/best.pt \
        --data <gate-cls> [--extra <file-or-dir> ...]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from urolens_ai.inference.input_gate import InputGate, check_exposure
from urolens_ai.inference.preprocessing import normalise
from urolens_ai.utils.exceptions import ImageValidationError

TARGET_SLIDE_PASS = 0.999


def score(gate: InputGate, path: Path) -> tuple[float, bool]:
    """Return (slide probability, passes exposure check) for one image file."""
    image = normalise(path.read_bytes())
    try:
        check_exposure(image)
        exposure_ok = True
    except ImageValidationError:
        exposure_ok = False
    return gate.slide_probability(image), exposure_ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="gate-cls root")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--extra", type=Path, nargs="*", default=[],
                        help="extra non-slide images to report individually")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = InputGate(model_path=str(args.weights), min_slide_prob=0.0)

    probs: dict[str, list[float]] = defaultdict(list)
    exposure_rejects: dict[str, int] = defaultdict(int)
    for cls in ("slide", "not_slide"):
        for path in sorted((args.data / args.split / cls).iterdir()):
            source = f"{cls}/{path.name.split('_', 1)[0]}"
            prob, exposure_ok = score(gate, path)
            probs[source].append(prob)
            exposure_rejects[source] += not exposure_ok

    uro = np.array(probs["slide/uro"])
    # Highest threshold that keeps >= TARGET_SLIDE_PASS of real slides.
    threshold = float(np.quantile(uro, 1 - TARGET_SLIDE_PASS, method="lower"))
    print(f"chosen threshold (>= {TARGET_SLIDE_PASS:.1%} UroLens pass): {threshold:.4f}\n")

    for thr in sorted({threshold, 0.1, 0.5}):
        print(f"threshold {thr:.4f}")
        for source, values in sorted(probs.items()):
            v = np.array(values)
            passed = (v >= thr).mean()
            label = "pass" if source.startswith("slide") else "pass (should be ~0)"
            print(f"  {source:22s} n={len(v):5d}  {label}: {passed:.4f}  "
                  f"exposure rejects: {exposure_rejects[source]}")
        print()

    for extra in args.extra:
        files = sorted(extra.iterdir()) if extra.is_dir() else [extra]
        for path in files:
            prob, exposure_ok = score(gate, path)
            print(f"extra {path.name}: slide_prob={prob:.4f} exposure_ok={exposure_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
