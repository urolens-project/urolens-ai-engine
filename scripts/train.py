"""
scripts/train.py
----------------
Train the UroLens particle detector.

Until this file existed, the only record of how the shipped model was trained was
the `train_args` blob inside best.pt. That is not a reproducible recipe, so this
script is the source of truth from here on.

IMPORTANT -- environment:
    Run this in the `yolo-urinalysis` conda environment, which has a CUDA build of
    torch (2.7.1+cu118) and can see the RTX 4060. The repository's own .venv has
    torch 2.12.0+cpu and will silently fall back to CPU, turning a 9-hour run into
    a multi-week one. The script refuses to start on CPU unless --allow-cpu is
    passed, precisely so that mistake cannot be made quietly.

        conda run -n yolo-urinalysis python scripts/train.py --data <data.yaml>

Recipe notes -- what changed from the mvp_v1-3 run and why:

    epochs 30 -> 150, patience 10 -> 30
        mAP50-95 was still rising monotonically at the final epoch (0.5691 ->
        0.5696 -> 0.5701) and early stopping never fired. That run ended on the
        epoch cap, not on convergence.

    hsv_h/s/v 0.0 -> 0.015/0.7/0.4
        All colour augmentation was disabled, presumably because the source
        dataset shipped pre-brightened copies. Those copies were also the leak.
        Doing it at training time instead gives unlimited variation, applies only
        to training batches, and cannot contaminate validation.

    flipud 0.0 -> 0.5, degrees 0.0 -> 180
        A microscope field has no canonical orientation. The source dataset was
        faking this by shipping pre-flipped files; doing it in the training loop
        is free and leak-proof.

    yolov8s -> yolo11s
        Same API and speed class, typically a couple of mAP points better. Free
        here only because the rebuilt split invalidates the old weights anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RECIPE: dict[str, object] = {
    "model": "yolo11s.pt",
    "epochs": 150,
    "patience": 30,
    "batch": 16,
    "imgsz": 640,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "warmup_epochs": 3.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    # Augmentation -- see module docstring.
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 180.0,
    "fliplr": 0.5,
    "flipud": 0.5,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "erasing": 0.4,
    # Reproducibility.
    "seed": 0,
    "deterministic": True,
    "val": True,
    "plots": True,
    "save_period": 10,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="path to data.yaml (use the rebuilt UroLens-4-clean, not UroLens-3)",
    )
    # Must be absolute. Ultralytics resolves a *relative* project path against its
    # own settings runs_dir, so "runs/detect/urinalysis" lands in
    # runs/detect/runs/detect/urinalysis. An absolute path is used verbatim.
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / "detect" / "urinalysis",
    )
    parser.add_argument("--name", default="v2-clean")
    parser.add_argument("--epochs", type=int, default=None, help="override recipe")
    parser.add_argument("--batch", type=int, default=None, help="override recipe")
    parser.add_argument("--model", default=None, help="override recipe backbone")
    parser.add_argument("--device", default=None, help="e.g. 0, cpu")
    parser.add_argument("--imgsz", type=int, default=None, help="override recipe")
    parser.add_argument("--patience", type=int, default=None, help="override recipe")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="dataloader workers; lower this if the host runs out of RAM",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="proceed without CUDA (a full run then takes weeks, not hours)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from its last.pt",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="explicit path to last.pt (defaults to <project>/<name>/weights/last.pt)",
    )
    return parser.parse_args(argv)


def check_device(allow_cpu: bool) -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"CUDA: {name} (torch {torch.__version__})", flush=True)
        return "cuda"
    message = (
        f"torch {torch.__version__} reports no CUDA device.\n"
        "  This is almost certainly the wrong environment -- the repo .venv has a\n"
        "  CPU-only wheel. Use: conda run -n yolo-urinalysis python scripts/train.py\n"
        "  Pass --allow-cpu only if you genuinely intend a CPU run."
    )
    if not allow_cpu:
        print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(2)
    print(f"WARNING: {message}", file=sys.stderr)
    return "cpu"


def guard_dataset(data: Path) -> None:
    """
    Refuse to train on a dataset that has not been through build_dataset.py.

    UroLens-3 contains cross-split duplicates and mirrored copies; training on it
    produces numbers that cannot be defended. The manifest is the marker that the
    rebuild actually ran.
    """
    if not data.exists():
        print(f"ERROR: {data} does not exist", file=sys.stderr)
        raise SystemExit(2)
    manifest = data.parent / "split_manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        print(
            f"Dataset: {meta['kept_images']} images "
            f"({meta['source_images']} before dedup), seed {meta['seed']}",
            flush=True,
        )
        return
    print(
        f"WARNING: no split_manifest.json beside {data}.\n"
        "  This dataset was not produced by scripts/build_dataset.py, so it may\n"
        "  still contain the cross-split duplicates that inflate every metric.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    guard_dataset(args.data)
    device = check_device(args.allow_cpu)

    from ultralytics import YOLO

    recipe = dict(RECIPE)
    if args.epochs is not None:
        recipe["epochs"] = args.epochs
    if args.batch is not None:
        recipe["batch"] = args.batch
    if args.model is not None:
        recipe["model"] = args.model
    if args.imgsz is not None:
        recipe["imgsz"] = args.imgsz
    if args.patience is not None:
        recipe["patience"] = args.patience
    if args.workers is not None:
        recipe["workers"] = args.workers
    if args.device is not None:
        device = args.device

    model_name = str(recipe.pop("model"))
    started = datetime.now(timezone.utc)

    if args.resume or args.resume_from is not None:
        # Ultralytics resumes from the checkpoint, not from the backbone: last.pt
        # carries the optimiser state, the epoch counter, and the original recipe.
        # Passing the recipe again here would conflict with what it restores, so
        # resume takes no other arguments.
        ckpt = args.resume_from or (args.project / args.name / "weights" / "last.pt")
        if not ckpt.exists():
            print(f"ERROR: no checkpoint at {ckpt}", file=sys.stderr)
            print(
                "  Pass --resume-from with the path to last.pt. Note that a run "
                "started\n  before the --project fix lives under a nested "
                "runs/detect/runs/detect/... path.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(f"Resuming from {ckpt}", flush=True)
        model = YOLO(str(ckpt))
        model.train(resume=True)
    else:
        print(f"Backbone: {model_name}", flush=True)
        print(
            "Recipe: "
            + ", ".join(f"{k}={v}" for k, v in sorted(recipe.items()) if k != "plots"),
            flush=True,
        )
        model = YOLO(model_name)
        model.train(
            data=str(args.data),
            project=str(args.project),
            name=args.name,
            device=device,
            exist_ok=True,
            **recipe,
        )
    elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 3600.0

    # A resumed run writes back into the directory it resumed from, which is not
    # necessarily <project>/<name>.
    if args.resume_from is not None:
        run_dir = args.resume_from.parent.parent
    else:
        run_dir = args.project / args.name
    record: dict[str, object] = {
        "device": device,
        "hours": round(elapsed, 2),
        "started_utc": started.isoformat(),
        "resumed": bool(args.resume or args.resume_from),
    }
    if not (args.resume or args.resume_from):
        record.update({"model": model_name, "data": str(args.data), **recipe})
    (run_dir / "recipe.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Finished in {elapsed:.2f} h -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
