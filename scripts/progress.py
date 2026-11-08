"""
scripts/progress.py
-------------------
Show a training run's curve, and compare it against a baseline run at the
same epoch. Reads results.csv, so it works while training is still going.

    python scripts/progress.py                      # default: v4-hires vs v2-clean
    python scripts/progress.py --last 20            # more rows
    python scripts/progress.py --run runs/detect/urinalysis/v4-hires
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_RUN = "runs/detect/urinalysis/v4-hires"
DEFAULT_BASE = "runs/detect/runs/detect/urinalysis/v2-clean"


def load(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    rows = [r for r in csv.DictReader(path.open()) if r["epoch"].strip().isdigit()]
    if not rows:
        return {}
    k50 = next(c for c in rows[0] if "mAP50(" in c)
    k95 = next(c for c in rows[0] if "mAP50-95" in c)
    kp = next(c for c in rows[0] if "precision" in c)
    kr = next(c for c in rows[0] if "recall" in c)
    return {
        int(r["epoch"]): {
            "mAP50": float(r[k50]),
            "mAP50-95": float(r[k95]),
            "P": float(r[kp]),
            "R": float(r[kr]),
        }
        for r in rows
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--baseline", default=DEFAULT_BASE)
    ap.add_argument("--last", type=int, default=12)
    args = ap.parse_args()

    run = load(Path(args.run) / "results.csv")
    base = load(Path(args.baseline) / "results.csv")
    if not run:
        raise SystemExit(f"no results.csv yet in {args.run}")

    done = max(run)
    print(f"run      : {args.run}")
    print(f"epochs   : {done}")
    print()
    print(f"{'ep':>4} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7} {'vs base':>9}")
    for e in sorted(run)[-args.last:]:
        m = run[e]
        gap = f"{m['mAP50'] - base[e]['mAP50']:+.4f}" if e in base else "-"
        print(
            f"{e:>4} {m['mAP50']:>8.4f} {m['mAP50-95']:>9.4f} "
            f"{m['P']:>7.3f} {m['R']:>7.3f} {gap:>9}"
        )

    best_ep = max(run, key=lambda e: run[e]["mAP50"])
    print()
    print(f"best     : epoch {best_ep} = {run[best_ep]['mAP50']:.4f} mAP50")
    if base:
        peak = max(base, key=lambda e: base[e]["mAP50"])
        print(f"baseline : {base[peak]['mAP50']:.4f} at epoch {peak} (val)")
        print("target   : 0.807 on the TEST split via scripts/evaluate.py")


if __name__ == "__main__":
    main()
