#!/usr/bin/env python3
"""Download and prepare the datasets the trainable stages need.

None of these datasets are bundled with the repo — they are large and carry their own
licences. This script fetches what can be fetched automatically and prints precise
instructions for the rest.

Datasets:
    ball   Tennis-ball detection, YOLO format, from Roboflow Universe. Needs a free
           Roboflow API key (https://app.roboflow.com -> Settings -> API key).
    court  Tennis-court keypoints, from Roboflow Universe. Same API key.
    thetis THETIS action-recognition clips (serve/forehand/backhand/volley). Requires
           manual download: the hosts gate it behind a request form.

Examples:
    python scripts/download_datasets.py --all --api-key YOUR_KEY
    python scripts/download_datasets.py --dataset ball --api-key YOUR_KEY
    python scripts/download_datasets.py --dataset thetis   # prints instructions
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: Roboflow Universe projects used as starting points. Override on the command line if
#: a project moves or you prefer a different one.
ROBOFLOW_DEFAULTS = {
    "ball": {
        "workspace": "viren-dhanwani",
        "project": "tennis-ball-detection",
        "version": 6,
        "format": "yolov8",
    },
    "court": {
        "workspace": "viren-dhanwani",
        "project": "tennis-court-keypoints",
        "version": 1,
        "format": "coco",
    },
}

THETIS_INSTRUCTIONS = """
THETIS — Three Dimensional Tennis Shots dataset
-----------------------------------------------
THETIS is not directly downloadable: the maintainers gate access behind a request.

  1. Visit the project page:
       http://thetis.image.ece.ntua.gr/

  2. Request access and download the RGB videos archive
     (the depth/skeleton archives are not needed here).

  3. Extract it so the layout looks like:

       data/thetis/
         forehand/
           p1_forehand_s1.avi
           ...
         backhand/
         service/            <- mapped to the "serve" label
         volley/             <- e.g. forehand_volley / backhand_volley

  4. Train the classifier:
       python scripts/train_stroke_classifier.py --data data/thetis --backend random_forest

     The trainer runs pose estimation over each clip, extracts the same features the
     live pipeline uses, and writes models/stroke_classifier.joblib.

Note on label mapping: THETIS ships finer-grained classes than the four this project
predicts. train_stroke_classifier.py folds them via its --label-map option; the
default mapping is printed by running that script with --show-label-map.
"""


def download_roboflow(
    dataset: str,
    api_key: str,
    output_dir: Path,
    workspace: str | None = None,
    project: str | None = None,
    version: int | None = None,
) -> int:
    """Download a Roboflow Universe dataset.

    Args:
        dataset: ``"ball"`` or ``"court"``.
        api_key: Roboflow API key.
        output_dir: Directory to download into.
        workspace: Override the default workspace slug.
        project: Override the default project slug.
        version: Override the default version number.

    Returns:
        Process-style status: 0 on success, 1 on failure.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            "error: the roboflow package is not installed.\n"
            "       pip install roboflow",
            file=sys.stderr,
        )
        return 1

    defaults = ROBOFLOW_DEFAULTS[dataset]
    workspace = workspace or defaults["workspace"]
    project = project or defaults["project"]
    version = version or defaults["version"]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset} dataset: {workspace}/{project} v{version} -> {output_dir}")

    try:
        roboflow = Roboflow(api_key=api_key)
        handle = roboflow.workspace(workspace).project(project).version(version)
        handle.download(defaults["format"], location=str(output_dir))
    except Exception as exc:  # noqa: BLE001 - surface whatever the SDK reports
        print(f"error: download failed: {exc}", file=sys.stderr)
        print(
            "\nIf the project has moved, browse https://universe.roboflow.com for an\n"
            "alternative and pass --workspace/--project/--version explicitly.",
            file=sys.stderr,
        )
        return 1

    print(f"done: {dataset} dataset is in {output_dir}")
    return 0


def show_thetis_instructions() -> int:
    """Print manual acquisition instructions for THETIS.

    Returns:
        Always 0.
    """
    print(THETIS_INSTRUCTIONS)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download the ball, court and stroke datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", choices=["ball", "court", "thetis"], help="which dataset to fetch"
    )
    parser.add_argument("--all", action="store_true", help="fetch every dataset")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY)",
    )
    parser.add_argument("--output", type=Path, default=Path("data"), help="output root")
    parser.add_argument("--workspace", help="override the Roboflow workspace slug")
    parser.add_argument("--project", help="override the Roboflow project slug")
    parser.add_argument("--version", type=int, help="override the Roboflow version")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the downloader.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    if not args.dataset and not args.all:
        build_parser().print_help()
        return 1

    targets = ["ball", "court", "thetis"] if args.all else [args.dataset]
    status = 0

    for target in targets:
        print(f"\n{'=' * 60}\n{target}\n{'=' * 60}")
        if target == "thetis":
            status |= show_thetis_instructions()
            continue

        if not args.api_key:
            print(
                "error: a Roboflow API key is required.\n"
                "       Pass --api-key or set ROBOFLOW_API_KEY.\n"
                "       Get one free at https://app.roboflow.com -> Settings -> API key",
                file=sys.stderr,
            )
            status = 1
            continue

        status |= download_roboflow(
            target,
            args.api_key,
            args.output / target,
            args.workspace,
            args.project,
            args.version,
        )

    return status


if __name__ == "__main__":
    raise SystemExit(main())
