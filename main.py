#!/usr/bin/env python3
"""CLI entry point for the Tennis Analysis System.

Examples:
    python main.py --input input_videos/match.mp4 --output output/
    python main.py --input clip.mp4 --output out/ --no-stroke --max-frames 200
    python main.py --input clip.mp4 --output out/ --dashboard
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from tennis_analysis.config import Config
from tennis_analysis.pipeline import PipelineResult, TennisAnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="tennis-analysis",
        description="Analyse a tennis video: track players and ball, map to court "
        "coordinates, compute speeds/distances and classify strokes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="input video file")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output"), help="output directory"
    )
    parser.add_argument(
        "--config", "-c", type=Path, default=Path("config.yaml"), help="config YAML/JSON"
    )

    stroke = parser.add_mutually_exclusive_group()
    stroke.add_argument(
        "--stroke", dest="stroke", action="store_true", default=None,
        help="force stroke classification on",
    )
    stroke.add_argument(
        "--no-stroke", dest="stroke", action="store_false",
        help="skip stroke classification (stages 1-5 only)",
    )

    parser.add_argument(
        "--dashboard", action="store_true",
        help="launch the Streamlit dashboard on the results when the run finishes",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="process at most this many frames (overrides config; useful for smoke tests)",
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="process every Nth frame (overrides config)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="enable debug logging"
    )
    return parser


def summarise(result: PipelineResult) -> str:
    """Render a short human-readable run summary."""
    lines = [
        "",
        "=" * 60,
        f"Video      : {result.metadata.path.name}",
        f"Resolution : {result.metadata.width}x{result.metadata.height} @ "
        f"{result.metadata.fps:.2f} fps",
        f"Frames     : {len(result.frame_indices)} processed "
        f"(effective {result.effective_fps:.2f} fps)",
        f"Players    : {result.player_ids or 'none identified'}",
        f"Shots      : {len(result.shots)}",
    ]

    if not result.summary.empty:
        lines.append("")
        lines.append(result.summary.to_string(index=False))

    if result.shots:
        counts: dict[str, int] = {}
        for shot in result.shots:
            counts[shot.stroke] = counts.get(shot.stroke, 0) + 1
        lines.append("")
        lines.append("Stroke distribution: " + ", ".join(
            f"{name}={count}" for name, count in sorted(counts.items())
        ))

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)

    lines.append("=" * 60)
    return "\n".join(lines)


def launch_dashboard(output_dir: Path) -> None:
    """Launch the Streamlit dashboard against a results directory.

    Args:
        output_dir: Directory containing the exported CSV/JSON artefacts.
    """
    app = Path(__file__).parent / "dashboard" / "app.py"
    if not app.exists():
        print(f"dashboard app not found at {app}", file=sys.stderr)
        return
    print(f"\nLaunching Streamlit dashboard for {output_dir} ...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), "--", "--results", str(output_dir)],
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline from the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        print(f"error: input video not found: {args.input}", file=sys.stderr)
        return 1

    try:
        config = Config.load(args.config if args.config.exists() else None)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: could not load config: {exc}", file=sys.stderr)
        return 1

    if not args.config.exists():
        logging.warning("config %s not found; using built-in defaults", args.config)

    if args.max_frames is not None:
        config.video.max_frames = args.max_frames
    if args.stride is not None:
        if args.stride < 1:
            print(f"error: --stride must be >= 1, got {args.stride}", file=sys.stderr)
            return 1
        config.video.frame_stride = args.stride

    try:
        pipeline = TennisAnalysisPipeline(config, enable_stroke=args.stroke)
        result = pipeline.run(args.input, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(summarise(result))
    print(f"\nArtefacts written to {args.output.resolve()}")

    if args.dashboard:
        launch_dashboard(args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
