"""
scripts/evaluate.py
-------------------
Evaluate a UroLens detector on what the product actually consumes.

mAP is the wrong headline metric for this system. The rule engine
(`urolens_ai.smart_diagnosis.rule_engine.run_rule_engine`) reads particle
*counts*, not boxes, and it compares them against fixed normal ranges. A model can
hold its mAP steady while systematically over- or under-counting a class, and that
shifts every diagnosis it feeds.

That failure is real here, not hypothetical: crystals run precision-poor (the model
over-detects, inflating gout) while erythrocytes run recall-poor (it under-detects,
deflating both glomerulonephritis and nephrolithiasis). Opposite error directions on
the three classes the rule engine reads, biasing three conditions two different ways
-- and completely invisible in an aggregate mAP number.

So this reports three layers:

    1. Detection metrics  -- per-class P/R/mAP50/mAP50-95, via Ultralytics.
    2. Count accuracy     -- per-class MAE and signed bias per field.
    3. Diagnosis accuracy -- LOW/MODERATE/HIGH agreement when the rule engine is
                             run on predicted counts vs. ground-truth counts.

Usage:
    python scripts/evaluate.py --weights runs/.../best.pt --data <data.yaml>
    python scripts/evaluate.py --weights ... --data ... --sweep-conf
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CONDITIONS = ("gout", "glomerulonephritis", "nephrolithiasis")
LEVELS = ("LOW", "MODERATE", "HIGH")

# The classes the rule engine actually reads. Tuning that ignores the rest is not
# a shortcut -- it is the point.
RULE_ENGINE_CLASSES = ("crystals", "erythrocytes", "urinary-casts")


def particle_key(class_name: str) -> str:
    """
    Convert a YOLO class name to the rule engine's particle key.

    Mirrors `urolens_ai.inference.postprocessing.map_detections`, which does the
    same dash-to-underscore normalisation on the live path.
    """
    return class_name.replace("-", "_")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def resolve_split(data_yaml: Path, split: str) -> tuple[Path, Path, list[str]]:
    """Resolve a split's image/label directories and the class-name list."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    key = {"train": "train", "valid": "val", "val": "val", "test": "test"}[split]
    rel = config.get(key)
    if rel is None:
        raise SystemExit(f"data.yaml has no '{key}' split")
    images = (data_yaml.parent / rel).resolve()
    if not images.is_dir():
        images = (data_yaml.parent / rel.replace("../", "")).resolve()
    labels = images.parent / "labels"
    if not images.is_dir():
        raise SystemExit(f"image directory not found: {images}")
    return images, labels, list(names)


def true_counts(label_path: Path, n_classes: int) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                index = int(parts[0])
            except ValueError:
                continue
            if 0 <= index < n_classes:
                counts[index] += 1
    return counts


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_detections(
    weights: Path,
    images: Path,
    labels: Path,
    names: list[str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    batch: int = 16,
    verbose: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    Run the detector over a split once and keep every detection above `conf`.

    Returns (detections, truth, files): detections[i] is a (k, 2) array of
    (class index, confidence) for image i; truth is the (n_images, n_classes)
    ground-truth count matrix. Counting at any stricter threshold is then free,
    see counts_at().
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    files = sorted(p.name for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"no images in {images}")

    detections: list[np.ndarray] = []
    truth = np.zeros((len(files), len(names)), dtype=np.int32)

    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        results = model.predict(
            [str(images / f) for f in chunk],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for offset, (filename, result) in enumerate(zip(chunk, results)):
            row = start + offset
            if result.boxes is not None and len(result.boxes):
                detections.append(
                    np.stack([result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()], 1)
                )
            else:
                detections.append(np.zeros((0, 2)))
            stem = filename.rsplit(".", 1)[0]
            for index, count in true_counts(labels / f"{stem}.txt", len(names)).items():
                truth[row, index] = count
        if verbose and start and start % (batch * 20) == 0:
            print(f"  {start}/{len(files)}", flush=True)
    return detections, truth, files


def counts_at(detections: list[np.ndarray], thresholds: np.ndarray) -> np.ndarray:
    """Count detections per image and class, keeping conf >= thresholds[class]."""
    predicted = np.zeros((len(detections), len(thresholds)), dtype=np.int32)
    for row, dets in enumerate(detections):
        if len(dets):
            cls = dets[:, 0].astype(int)
            keep = dets[:, 1] >= thresholds[cls]
            np.add.at(predicted[row], cls[keep], 1)
    return predicted


def predict_counts(
    weights: Path,
    images: Path,
    labels: Path,
    names: list[str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    batch: int = 16,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run the detector over a split and return (predicted, truth) count matrices.

    Both are (n_images, n_classes) integer arrays aligned by row.
    """
    detections, truth, files = predict_detections(
        weights, images, labels, names, conf, iou, imgsz, device, batch, verbose
    )
    return counts_at(detections, np.full(len(names), conf)), truth, files


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def detection_metrics(
    weights: Path, data_yaml: Path, split: str, imgsz: int, conf: float, iou: float,
    device: str,
) -> dict[str, dict[str, float]]:
    """Per-class P/R/mAP from Ultralytics' own validator."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="val" if split == "valid" else split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        plots=False,
    )
    out: dict[str, dict[str, float]] = {}
    box = metrics.box
    for position, class_index in enumerate(box.ap_class_index):
        out[metrics.names[int(class_index)]] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "map50": float(box.ap50[position]),
            "map50_95": float(box.ap[position]),
        }
    out["__all__"] = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
    return out


def count_metrics(
    predicted: np.ndarray, truth: np.ndarray, names: list[str]
) -> dict[str, dict[str, float]]:
    """Per-class count error -- what the rule engine's inputs are actually worth."""
    out: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        p = predicted[:, index].astype(float)
        t = truth[:, index].astype(float)
        total = t.sum()
        out[name] = {
            "true_per_image": float(t.mean()),
            "pred_per_image": float(p.mean()),
            "mae": float(np.abs(p - t).mean()),
            "bias": float(p.mean() - t.mean()),
            "bias_pct": float((p.sum() - total) / total * 100.0) if total else 0.0,
        }
    return out


def quiet_engine_logging() -> None:
    """
    Silence the rule engine's per-call DEBUG logging.

    get_logger() attaches a handler and sets DEBUG on each module logger
    individually, so raising the level on the parent does not propagate. Over
    thousands of images the output is unreadable otherwise.
    """
    import logging

    for name in list(logging.root.manager.loggerDict):
        if name.startswith("urolens_ai"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _cohen_kappa(confusion: collections.Counter, levels: tuple[str, ...]) -> float:
    """
    Chance-corrected agreement.

    Raw agreement is close to meaningless on this data: the HIGH band is nearly
    unreachable under the shipped config, so both sides say LOW almost always and
    a coin-flip model would still score in the high nineties. Kappa removes that
    floor. Returns 0.0 when one rater is entirely constant (kappa undefined).
    """
    total = sum(confusion.values())
    if not total:
        return 0.0
    observed = sum(n for (e, a), n in confusion.items() if e == a) / total
    expected = 0.0
    for level in levels:
        true_marginal = sum(n for (e, _a), n in confusion.items() if e == level) / total
        pred_marginal = sum(n for (_e, a), n in confusion.items() if a == level) / total
        expected += true_marginal * pred_marginal
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def diagnosis_agreement(
    predicted: np.ndarray, truth: np.ndarray, names: list[str]
) -> dict[str, dict]:
    """
    Run the real rule engine on both count sets and compare the levels.

    This is the end-to-end number: how often the product reaches the same
    conclusion from the model's counts as it would from perfect counts.

    Raw agreement alone is reported alongside the ground-truth level distribution
    and per-level recall, because on this config the raw figure flatters badly --
    when ~97% of fields are LOW under perfect counts, predicting LOW every time
    already scores ~97%. What matters is whether the non-LOW cases survive.
    """
    from urolens_ai.smart_diagnosis.rule_engine import run_rule_engine

    quiet_engine_logging()

    agree: collections.Counter = collections.Counter()
    confusion: dict[str, collections.Counter] = {
        c: collections.Counter() for c in CONDITIONS
    }
    for row in range(predicted.shape[0]):
        pred_dict = {
            particle_key(name): int(predicted[row, i]) for i, name in enumerate(names)
        }
        true_dict = {
            particle_key(name): int(truth[row, i]) for i, name in enumerate(names)
        }
        pred_out = run_rule_engine(pred_dict)
        true_out = run_rule_engine(true_dict)
        for condition in CONDITIONS:
            expected = getattr(true_out, condition).level.value
            actual = getattr(pred_out, condition).level.value
            confusion[condition][(expected, actual)] += 1
            if expected == actual:
                agree[condition] += 1

    total = predicted.shape[0]
    out: dict[str, dict] = {}
    for condition in CONDITIONS:
        counter = confusion[condition]
        truth_dist = {
            level: sum(n for (e, _a), n in counter.items() if e == level)
            for level in LEVELS
        }
        pred_dist = {
            level: sum(n for (_e, a), n in counter.items() if a == level)
            for level in LEVELS
        }
        per_level = {}
        for level in LEVELS:
            support = truth_dist[level]
            hits = counter.get((level, level), 0)
            per_level[level] = {
                "support": support,
                "recall": (hits / support) if support else None,
            }
        out[condition] = {
            "agreement": agree[condition] / total if total else 0.0,
            "kappa": _cohen_kappa(counter, LEVELS),
            "majority_baseline": (max(truth_dist.values()) / total) if total else 0.0,
            "n": total,
            "truth_distribution": truth_dist,
            "pred_distribution": pred_dist,
            "per_level": per_level,
            "confusion": {f"{e}->{a}": n for (e, a), n in counter.items()},
        }
    return out


def print_report(
    detection: dict[str, dict[str, float]] | None,
    counts: dict[str, dict[str, float]],
    diagnosis: dict[str, dict],
    names: list[str],
    conf: float,
) -> None:
    print(f"\n{'=' * 78}")
    print(f"DETECTION METRICS (mAP pass at conf={conf:g})")
    print("=" * 78)
    if detection:
        header = f"{'class':<24}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}"
        print(header)
        for name in names:
            row = detection.get(name)
            if not row:
                print(f"{name:<24}{'--- no instances in split ---':>35}")
                continue
            marker = " *" if name in RULE_ENGINE_CLASSES else "  "
            print(
                f"{name:<22}{marker}{row['precision']:>8.3f}{row['recall']:>8.3f}"
                f"{row['map50']:>9.3f}{row['map50_95']:>10.3f}"
            )
        overall = detection["__all__"]
        print(
            f"{'ALL':<24}{overall['precision']:>8.3f}{overall['recall']:>8.3f}"
            f"{overall['map50']:>9.3f}{overall['map50_95']:>10.3f}"
        )
        print("  * consumed by the rule engine")

    print(f"\n{'=' * 78}")
    print("COUNT ACCURACY -- what the rule engine actually receives")
    print("=" * 78)
    print(
        f"{'class':<24}{'true/img':>10}{'pred/img':>10}{'MAE':>8}{'bias':>9}{'bias %':>9}"
    )
    for name in names:
        row = counts[name]
        marker = " *" if name in RULE_ENGINE_CLASSES else "  "
        print(
            f"{name:<22}{marker}{row['true_per_image']:>10.2f}{row['pred_per_image']:>10.2f}"
            f"{row['mae']:>8.2f}{row['bias']:>+9.2f}{row['bias_pct']:>+8.1f}%"
        )

    print(f"\n{'=' * 78}")
    print("DIAGNOSIS AGREEMENT -- model counts vs ground-truth counts")
    print("=" * 78)
    print(
        "Read `agreement` against `always-LOW` on the same line. If they are close,\n"
        "the model is not agreeing so much as the score bands are collapsed --\n"
        "kappa is the chance-corrected figure and is the one to quote.\n"
    )
    for condition in CONDITIONS:
        row = diagnosis[condition]
        print(
            f"{condition:<24}agreement {row['agreement'] * 100:>5.1f}%   "
            f"always-LOW {row['majority_baseline'] * 100:>5.1f}%   "
            f"kappa {row['kappa']:>5.2f}   (n={row['n']})"
        )
        dist = " ".join(
            f"{level}={row['truth_distribution'][level]}" for level in LEVELS
        )
        print(f"{'':<24}ground truth: {dist}")
        for level in LEVELS:
            entry = row["per_level"][level]
            if not entry["support"]:
                print(
                    f"{'':<24}  {level:<9} never occurs under perfect counts "
                    f"-- this band is unreachable"
                )
            else:
                print(
                    f"{'':<24}  {level:<9} recall {entry['recall'] * 100:>5.1f}% "
                    f"of {entry['support']}"
                )
        disagreements = {
            k: v
            for k, v in row["confusion"].items()
            if k.split("->")[0] != k.split("->")[1]
        }
        if disagreements:
            detail = ", ".join(
                f"{k} x{v}"
                for k, v in sorted(disagreements.items(), key=lambda kv: -kv[1])
            )
            print(f"{'':<24}  errors: {detail}")
        print()


# ---------------------------------------------------------------------------
# Confidence sweep
# ---------------------------------------------------------------------------


def sweep(
    weights: Path, images: Path, labels: Path, names: list[str], args: argparse.Namespace
) -> list[dict]:
    """
    Choose a confidence threshold by count bias, not by mAP.

    Note the live inconsistency this is meant to settle: .env sets
    INFERENCE_CONF_THRESHOLD=0.25 while the default in
    src/urolens_ai/inference/yolo_engine.py is 0.45. Every particle count -- and
    therefore every diagnosis -- depends on which one is in effect.
    """
    rows: list[dict] = []
    for conf in [round(c, 2) for c in np.arange(0.10, 0.71, 0.05)]:
        predicted, truth, _ = predict_counts(
            weights, images, labels, names, conf, args.iou, args.imgsz,
            args.device, verbose=False,
        )
        counts = count_metrics(predicted, truth, names)
        tracked = [counts[n] for n in RULE_ENGINE_CLASSES if n in counts]
        row = {
            "conf": conf,
            "mean_abs_bias_pct": float(
                np.mean([abs(c["bias_pct"]) for c in tracked])
            ),
            "mean_mae": float(np.mean([c["mae"] for c in tracked])),
        }
        for name in RULE_ENGINE_CLASSES:
            if name in counts:
                row[f"{name}_bias_pct"] = counts[name]["bias_pct"]
        rows.append(row)
        print(
            f"  conf={conf:.2f}  mean|bias|={row['mean_abs_bias_pct']:>6.1f}%  "
            f"mean MAE={row['mean_mae']:.2f}",
            flush=True,
        )
    best = min(rows, key=lambda r: r["mean_abs_bias_pct"])
    print(
        f"\nLowest count bias on the rule-engine classes at conf={best['conf']:.2f} "
        f"(mean |bias| {best['mean_abs_bias_pct']:.1f}%)"
    )
    return rows


PER_CLASS_GRID = [round(c, 2) for c in np.arange(0.10, 0.81, 0.05)]

# A class keeps the uniform baseline unless moving improves its count MAE by at
# least this much. Smaller gains are within what tuning on one split can produce
# by chance, and would not survive to the test split.
PER_CLASS_MIN_GAIN = 0.05


def sweep_per_class(
    weights: Path, images: Path, labels: Path, names: list[str], args: argparse.Namespace
) -> dict[str, float]:
    """
    Choose one confidence threshold per class by per-image count error.

    A single threshold forces one trade-off on every class: raising it to quiet a
    noisy class (sperm-cells) also costs recall on the classes the rule engine
    reads (urinary-casts lose ~10 recall points going 0.35 -> 0.5). Counts per
    class depend only on that class's threshold, so each one can be chosen
    independently: the value minimising mean absolute count error per image,
    which is the error a MedTech sees on each result.

    The detector runs once at the bottom of the grid; every candidate is then
    a re-count. Diagnosis agreement is reported against the uniform
    --conf baseline so the choice can be rejected if it hurts the product.
    """
    detections, truth, _ = predict_detections(
        weights, images, labels, names, PER_CLASS_GRID[0], args.iou, args.imgsz,
        args.device, verbose=False,
    )
    baseline = np.full(len(names), args.conf)
    chosen = baseline.copy()

    print(f"\n{'class':<24}{'MAE@' + format(args.conf, 'g'):>10}{'best conf':>11}{'MAE':>8}"
          f"{'bias%@base':>12}{'bias%@best':>12}")
    for index, name in enumerate(names):
        maes = []
        for conf in PER_CLASS_GRID:
            thresholds = baseline.copy()
            thresholds[index] = conf
            predicted = counts_at(detections, thresholds)
            maes.append(np.abs(predicted[:, index] - truth[:, index]).mean())
        base_mae = np.abs(counts_at(detections, baseline)[:, index] - truth[:, index]).mean()
        if min(maes) <= base_mae * (1 - PER_CLASS_MIN_GAIN):
            chosen[index] = PER_CLASS_GRID[int(np.argmin(maes))]
        base_counts = count_metrics(counts_at(detections, baseline), truth, names)[name]
        best_counts = count_metrics(counts_at(detections, chosen), truth, names)[name]
        print(f"{name:<24}{base_counts['mae']:>10.3f}{chosen[index]:>11.2f}"
              f"{best_counts['mae']:>8.3f}{base_counts['bias_pct']:>12.1f}"
              f"{best_counts['bias_pct']:>12.1f}")

    print("\nDiagnosis agreement (kappa), uniform baseline vs per-class:")
    base_dx = diagnosis_agreement(counts_at(detections, baseline), truth, names)
    best_dx = diagnosis_agreement(counts_at(detections, chosen), truth, names)
    for condition in CONDITIONS:
        print(f"  {condition:<20} {base_dx[condition]['kappa']:.4f} -> "
              f"{best_dx[condition]['kappa']:.4f}   agreement "
              f"{base_dx[condition]['agreement']:.4f} -> {best_dx[condition]['agreement']:.4f}")
    return {name: float(chosen[i]) for i, name in enumerate(names)}


def load_thresholds(path: Path, names: list[str], default: float) -> np.ndarray:
    """Per-class threshold vector from a thresholds.yaml; missing classes use `default`."""
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    per_class = config.get("per_class", {})
    return np.array([float(per_class.get(name, default)) for name in names])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="path to data.yaml")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="deployment threshold, used for COUNTS and diagnosis agreement",
    )
    parser.add_argument(
        "--map-conf",
        type=float,
        default=0.001,
        help=(
            "threshold for the mAP pass. Must stay near zero: average precision is "
            "the area under the full precision-recall curve, so a deployment-level "
            "threshold truncates the low-confidence tail and understates mAP. Only "
            "change this to reproduce someone else's differently-measured number."
        ),
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--json", type=Path, default=None, help="write report as JSON")
    parser.add_argument(
        "--sweep-conf",
        action="store_true",
        help="sweep confidence and pick the threshold minimising count bias",
    )
    parser.add_argument(
        "--sweep-per-class",
        action="store_true",
        help="choose one threshold per class by count MAE (run on --split valid)",
    )
    parser.add_argument(
        "--thresholds-out",
        type=Path,
        default=None,
        help="with --sweep-per-class: write the chosen thresholds as YAML",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="per-class thresholds.yaml for COUNTS and diagnosis (overrides --conf per class)",
    )
    parser.add_argument(
        "--skip-detection-metrics",
        action="store_true",
        help="skip the Ultralytics validator pass (counts and diagnosis only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault(
        "RULE_ENGINE_CONFIG_PATH",
        str(REPO_ROOT / "src/urolens_ai/smart_diagnosis/config.yaml"),
    )
    images, labels, names = resolve_split(args.data, args.split)
    print(f"Split: {args.split} ({images})", flush=True)

    if args.sweep_conf:
        rows = sweep(args.weights, images, labels, names, args)
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0

    if args.sweep_per_class:
        if args.split == "test":
            print("warning: choosing thresholds on the test split leaks into the "
                  "reported result -- use --split valid", file=sys.stderr)
        chosen = sweep_per_class(args.weights, images, labels, names, args)
        if args.thresholds_out:
            args.thresholds_out.write_text(
                "# Per-class confidence thresholds, chosen by scripts/evaluate.py "
                f"--sweep-per-class on the {args.split} split.\n"
                "# Re-run whenever the model weights change.\n"
                + yaml.safe_dump({"per_class": chosen}, sort_keys=False),
                encoding="utf-8",
            )
            print(f"\nWrote {args.thresholds_out}")
        return 0

    detection = None
    if not args.skip_detection_metrics:
        print(f"Running detection metrics (conf={args.map_conf:g})", flush=True)
        detection = detection_metrics(
            args.weights, args.data, args.split, args.imgsz, args.map_conf, args.iou,
            args.device,
        )

    print("Predicting counts", flush=True)
    if args.thresholds:
        thresholds = load_thresholds(args.thresholds, names, args.conf)
        detections, truth, files = predict_detections(
            args.weights, images, labels, names, float(thresholds.min()), args.iou,
            args.imgsz, args.device,
        )
        predicted = counts_at(detections, thresholds)
    else:
        predicted, truth, files = predict_counts(
            args.weights, images, labels, names, args.conf, args.iou, args.imgsz, args.device
        )
    counts = count_metrics(predicted, truth, names)
    diagnosis = diagnosis_agreement(predicted, truth, names)
    print_report(detection, counts, diagnosis, names, args.map_conf)

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "weights": str(args.weights),
                    "data": str(args.data),
                    "split": args.split,
                    "conf": args.conf,
                    "iou": args.iou,
                    "n_images": len(files),
                    "detection": detection,
                    "counts": counts,
                    "diagnosis": diagnosis,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
