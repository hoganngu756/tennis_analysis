#!/usr/bin/env python3
"""Fine-tune a YOLO11 model for tennis-ball detection.

The dataset must be in Ultralytics YOLO format, i.e. a ``data.yaml`` alongside
``train/`` and ``valid/`` image and label directories — which is what
``scripts/download_datasets.py --dataset ball`` produces.

Examples:
    python scripts/train_ball_detector.py --data data/ball/data.yaml
    python scripts/train_ball_detector.py --data data/ball/data.yaml \\
        --model yolo11s.pt --epochs 100 --imgsz 1280
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11 on a tennis-ball dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="path to the dataset data.yaml"
    )
    parser.add_argument(
        "--model", default="yolo11n.pt", help="base checkpoint to fine-tune from"
    )
    parser.add_argument("--epochs", type=int, default=60, help="training epochs")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="training image size; raise to 1280 if the ball is only a few pixels wide",
    )
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument(
        "--device", default=None, help="cuda device, 'mps', or 'cpu' (auto if unset)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ball_yolo11n.pt"),
        help="where to copy the best weights when training finishes",
    )
    parser.add_argument("--name", default="ball_detector", help="Ultralytics run name")
    parser.add_argument(
        "--patience", type=int, default=20, help="early-stopping patience in epochs"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fine-tune the ball detector.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    args = build_parser().parse_args(argv)

    if not args.data.exists():
        print(
            f"error: dataset config not found: {args.data}\n"
            f"       Fetch one with: python scripts/download_datasets.py "
            f"--dataset ball --api-key YOUR_KEY",
            file=sys.stderr,
        )
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: ultralytics is not installed. pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    print(f"Fine-tuning {args.model} on {args.data} for {args.epochs} epochs")
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # The ball is tiny and nearly always in motion. Aggressive geometric
        # augmentation hurts more than it helps here, while mosaic (which pastes four
        # images together) is genuinely useful for small-object recall.
        mosaic=1.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"error: training finished but {best} is missing", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print(f"\nBest weights copied to {args.output}")
    print(f"Point config.yaml at it:  ball.model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
