#!/usr/bin/env python3
"""Train the stroke classifier on labelled stroke clips (THETIS by default).

Expected layout — one directory per source class, each holding video clips:

    data/thetis/
      forehand_flat/     *.avi
      forehand_slice/    *.avi
      backhand/          *.avi
      service_flat/      *.avi
      forehand_volley/   *.avi
      ...

Source class names are folded into the four labels the pipeline predicts
(serve / forehand / backhand / volley) by the mapping below. Run with
``--show-label-map`` to print it, and override with ``--label-map`` if your copy of
the dataset uses different directory names.

Each clip is run through YOLO11-pose, converted into the same per-frame features the
live pipeline uses, and handed to the chosen backend.

Examples:
    python scripts/train_stroke_classifier.py --data data/thetis
    python scripts/train_stroke_classifier.py --data data/thetis --backend cnn1d
    python scripts/train_stroke_classifier.py --show-label-map
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis_analysis.stroke_classification.classifier import (  # noqa: E402
    build_classifier,
    write_label_metadata,
)
from tennis_analysis.stroke_classification.features import NUM_POSE_KEYPOINTS  # noqa: E402

#: Ordered patterns matched against the lower-cased source directory name. The first
#: match wins, so the volley patterns must precede the plain forehand/backhand ones.
DEFAULT_LABEL_MAP: list[tuple[str, str]] = [
    (r"volley", "volley"),
    (r"service|serve", "serve"),
    (r"smash", "serve"),
    (r"forehand", "forehand"),
    (r"backhand", "backhand"),
]

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def map_label(directory_name: str, patterns: list[tuple[str, str]]) -> str | None:
    """Fold a source class directory name into one of the pipeline's stroke labels.

    Args:
        directory_name: Source class directory name.
        patterns: Ordered ``(regex, label)`` pairs; the first match wins.

    Returns:
        The mapped label, or ``None`` if nothing matched.
    """
    name = directory_name.lower()
    for pattern, label in patterns:
        if re.search(pattern, name):
            return label
    return None


def extract_clip_poses(
    path: Path, estimator, max_frames: int = 120
) -> np.ndarray | None:
    """Run pose estimation over every frame of a clip.

    THETIS clips are tightly framed on a single player, so pose runs on the full frame
    rather than on a tracked bounding box.

    Args:
        path: Video clip path.
        estimator: A loaded Ultralytics pose model.
        max_frames: Cap on frames read per clip.

    Returns:
        ``(T, 17, 3)`` pose array, or ``None`` if no pose was found in the clip.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None

    poses: list[np.ndarray] = []
    try:
        while len(poses) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            results = estimator.predict(frame, verbose=False)
            if not results:
                continue
            keypoints = results[0].keypoints
            if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
                continue
            data = keypoints.data.cpu().numpy()
            best = int(np.argmax(data[:, :, 2].mean(axis=1)))
            pose = data[best].astype(np.float64)
            if pose.shape == (NUM_POSE_KEYPOINTS, 3):
                poses.append(pose)
    finally:
        capture.release()

    return np.stack(poses) if poses else None


def load_dataset(
    root: Path, patterns: list[tuple[str, str]], pose_model: str, max_clips: int | None
) -> tuple[list[np.ndarray], list[str]]:
    """Load and featurise every clip under ``root``.

    Args:
        root: Dataset root containing one directory per source class.
        patterns: Label mapping patterns.
        pose_model: Ultralytics pose checkpoint.
        max_clips: Cap on clips per source class (useful for a quick trial run).

    Returns:
        ``(windows, labels)`` — pose windows and their mapped stroke labels.
    """
    from ultralytics import YOLO

    estimator = YOLO(pose_model)
    windows: list[np.ndarray] = []
    labels: list[str] = []
    skipped = 0

    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        label = map_label(directory.name, patterns)
        if label is None:
            print(f"  skipping unmapped directory: {directory.name}")
            continue

        clips = sorted(
            p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES
        )
        if max_clips is not None:
            clips = clips[:max_clips]

        print(f"  {directory.name} -> {label}: {len(clips)} clips")
        for clip in clips:
            window = extract_clip_poses(clip, estimator)
            if window is None or len(window) < 2:
                skipped += 1
                continue
            windows.append(window)
            labels.append(label)

    if skipped:
        print(f"  ({skipped} clips skipped — no usable pose detected)")
    return windows, labels


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train the stroke classifier on labelled stroke clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, help="dataset root (one dir per class)")
    parser.add_argument(
        "--backend",
        choices=["random_forest", "cnn1d"],
        default="random_forest",
        help="classifier backend",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output model path (defaults by backend)",
    )
    parser.add_argument(
        "--pose-model", default="yolo11n-pose.pt", help="Ultralytics pose checkpoint"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="held-out validation fraction"
    )
    parser.add_argument("--seed", type=int, default=0, help="split/training seed")
    parser.add_argument(
        "--max-clips", type=int, default=None, help="cap clips per source class"
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="JSON file with a {regex: label} mapping, overriding the default",
    )
    parser.add_argument(
        "--show-label-map", action="store_true", help="print the default mapping and exit"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="cache extracted poses to this .npz to make re-training instant",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train the stroke classifier.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    args = build_parser().parse_args(argv)

    if args.show_label_map:
        print("Default source-class -> stroke-label mapping (first match wins):\n")
        for pattern, label in DEFAULT_LABEL_MAP:
            print(f"  /{pattern}/  ->  {label}")
        return 0

    if args.data is None:
        print("error: --data is required (or use --show-label-map)", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(
            f"error: dataset not found: {args.data}\n"
            f"       See: python scripts/download_datasets.py --dataset thetis",
            file=sys.stderr,
        )
        return 1

    patterns = DEFAULT_LABEL_MAP
    if args.label_map:
        patterns = list(json.loads(args.label_map.read_text()).items())

    output = args.output or Path(
        "models/stroke_classifier.joblib"
        if args.backend == "random_forest"
        else "models/stroke_classifier.pt"
    )

    windows: list[np.ndarray]
    labels: list[str]
    if args.cache and args.cache.exists():
        print(f"Loading cached poses from {args.cache}")
        payload = np.load(args.cache, allow_pickle=True)
        windows = list(payload["windows"])
        labels = list(payload["labels"])
    else:
        print(f"Extracting poses from {args.data} ...")
        windows, labels = load_dataset(
            args.data, patterns, args.pose_model, args.max_clips
        )
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.cache,
                windows=np.array(windows, dtype=object),
                labels=np.array(labels),
            )
            print(f"Cached poses to {args.cache}")

    if not windows:
        print("error: no usable clips found; check the dataset layout and label map",
              file=sys.stderr)
        return 1

    counts = {label: labels.count(label) for label in sorted(set(labels))}
    print(f"\nDataset: {len(windows)} clips  {counts}")
    if len(counts) < 2:
        print("error: need at least 2 distinct labels to train", file=sys.stderr)
        return 1

    # Stratified split, so every class is represented on both sides.
    rng = np.random.default_rng(args.seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in counts:
        indices = [i for i, value in enumerate(labels) if value == label]
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * args.val_fraction)) if len(indices) > 1 else 0
        val_indices.extend(indices[:cut])
        train_indices.extend(indices[cut:])

    train_windows = [windows[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_windows = [windows[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"Split: {len(train_windows)} train / {len(val_windows)} val")
    print(f"Training {args.backend} backend ...")

    classifier = build_classifier(args.backend)
    metrics = classifier.fit(train_windows, train_labels)

    if val_windows:
        metrics["val_accuracy"] = classifier.score(val_windows, val_labels)
        _print_confusion(classifier, val_windows, val_labels)

    classifier.save(output)
    write_label_metadata(output, classifier.labels, metrics)

    print("\nMetrics: " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print(f"Model written to {output}")
    print(f"Point config.yaml at it:  stroke.classifier: {output}")
    return 0


def _print_confusion(classifier, windows: list[np.ndarray], labels: list[str]) -> None:
    """Print a plain-text confusion matrix over the validation set."""
    names = sorted(set(labels) | set(classifier.labels))
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for window, truth in zip(windows, labels):
        predicted, _ = classifier.predict(window)
        matrix[index[truth], index[predicted]] += 1

    width = max(len(name) for name in names) + 2
    print("\nValidation confusion (rows = truth, cols = predicted):")
    print(" " * width + "".join(name[:6].rjust(8) for name in names))
    for i, name in enumerate(names):
        print(name.ljust(width) + "".join(str(value).rjust(8) for value in matrix[i]))


if __name__ == "__main__":
    raise SystemExit(main())
