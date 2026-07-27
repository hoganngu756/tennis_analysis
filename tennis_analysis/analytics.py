"""Stage 5 & 7 — speed/distance computation and shot-log export.

Everything here operates on *court-space metres* produced by the homography, so the
outputs are real-world quantities rather than pixel rates. Functions are pure and take
explicit inputs, which is what makes them straightforward to unit test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .types import Shot

#: Metres per second -> kilometres per hour.
MPS_TO_KMH = 3.6

#: A court-space position, or ``None`` for a frame where the entity was not located.
Position = tuple[float, float] | None


def displacement(a: Position, b: Position) -> float | None:
    """Euclidean distance in metres between two court-space positions.

    Args:
        a: First position, or ``None``.
        b: Second position, or ``None``.

    Returns:
        The distance in metres, or ``None`` if either position is missing or non-finite.
    """
    if a is None or b is None:
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def speed_kmh(distance_m: float, elapsed_s: float) -> float:
    """Convert a distance and elapsed time into a speed in km/h.

    Args:
        distance_m: Distance covered, in metres.
        elapsed_s: Time taken, in seconds. Must be > 0.

    Returns:
        Speed in kilometres per hour.

    Raises:
        ValueError: If ``elapsed_s`` is not positive.
    """
    if elapsed_s <= 0:
        raise ValueError(f"elapsed_s must be > 0, got {elapsed_s}")
    return (distance_m / elapsed_s) * MPS_TO_KMH


def compute_speed_series(
    positions: list[Position],
    fps: float,
    window: int = 5,
    max_speed_kmh: float | None = None,
) -> list[float | None]:
    """Compute a smoothed per-frame speed series from court-space positions.

    Speed at frame ``i`` is measured over a backward window: the displacement from the
    nearest valid position at or before ``i - window`` to the position at ``i``,
    divided by the actual elapsed time between those two frames. Measuring over a span
    rather than adjacent frames suppresses the jitter that single-frame differencing
    amplifies, and using the *actual* frame gap keeps the result correct across
    detection dropouts.

    Args:
        positions: One court-space position per frame, ``None`` where unknown.
        fps: Frame rate of the position series (i.e. already stride-adjusted).
        window: Number of frames to measure displacement over. Must be >= 1.
        max_speed_kmh: Speeds above this are treated as tracking noise and returned as
            ``None``. Pass ``None`` to disable clipping.

    Returns:
        One speed in km/h per frame, ``None`` where it could not be computed.

    Raises:
        ValueError: If ``fps`` <= 0 or ``window`` < 1.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    speeds: list[float | None] = [None] * len(positions)
    for i, current in enumerate(positions):
        if current is None:
            continue
        reference = _find_reference_index(positions, i, window)
        if reference is None:
            continue

        distance = displacement(positions[reference], current)
        if distance is None:
            continue

        elapsed = (i - reference) / fps
        value = speed_kmh(distance, elapsed)
        if max_speed_kmh is not None and value > max_speed_kmh:
            continue
        speeds[i] = value
    return speeds


def _find_reference_index(positions: list[Position], index: int, window: int) -> int | None:
    """Find the nearest valid position index at or before ``index - window``.

    Falls back to searching forward from that point toward ``index`` so a short gap
    shortens the measurement window rather than voiding the sample entirely.
    """
    target = index - window
    if target < 0:
        target = 0
    for candidate in range(target, index):
        if positions[candidate] is not None:
            return candidate
    return None


def cumulative_distance(
    positions: list[Position],
    fps: float,
    max_speed_kmh: float | None = None,
) -> list[float]:
    """Accumulate total distance covered, in metres, across a position series.

    Args:
        positions: One court-space position per frame, ``None`` where unknown.
        fps: Frame rate of the position series.
        max_speed_kmh: Steps implying a speed above this are treated as ID switches or
            detection noise and excluded from the total. ``None`` disables the filter.

    Returns:
        Cumulative distance in metres at each frame; the series is non-decreasing and
        holds its previous value across gaps.

    Raises:
        ValueError: If ``fps`` <= 0.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    max_step_m = None if max_speed_kmh is None else (max_speed_kmh / MPS_TO_KMH) / fps

    totals: list[float] = []
    running = 0.0
    previous: Position = None
    for current in positions:
        step = displacement(previous, current)
        if step is not None and (max_step_m is None or step <= max_step_m):
            running += step
        if current is not None:
            previous = current
        totals.append(running)
    return totals


def summarise_players(
    distances: dict[int, list[float]],
    speeds: dict[int, list[float | None]],
) -> pd.DataFrame:
    """Build a per-player summary of distance covered and speed statistics.

    Args:
        distances: Cumulative-distance series per player track ID.
        speeds: Per-frame speed series per player track ID.

    Returns:
        A DataFrame with one row per player: ``player``, ``total_distance_m``,
        ``avg_speed_kmh``, ``max_speed_kmh``.
    """
    rows = []
    for player_id in sorted(distances):
        series = [s for s in speeds.get(player_id, []) if s is not None]
        rows.append(
            {
                "player": player_id,
                "total_distance_m": round(distances[player_id][-1], 2)
                if distances[player_id]
                else 0.0,
                "avg_speed_kmh": round(float(np.mean(series)), 2) if series else 0.0,
                "max_speed_kmh": round(float(np.max(series)), 2) if series else 0.0,
            }
        )
    return pd.DataFrame(rows)


def export_shots(
    shots: list[Shot],
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write the per-shot log to CSV and/or JSON.

    Args:
        shots: Detected and classified shots.
        csv_path: Destination CSV path, or ``None`` to skip.
        json_path: Destination JSON path, or ``None`` to skip.

    Returns:
        The shot log as a DataFrame (empty, with the correct columns, if no shots).
    """
    columns = [
        "frame",
        "timestamp",
        "player",
        "stroke",
        "stroke_confidence",
        "ball_speed_kmh",
        "player_speed_kmh",
    ]
    frame = pd.DataFrame([shot.to_dict() for shot in shots], columns=columns)

    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([shot.to_dict() for shot in shots], indent=2))

    return frame


def export_tracks(
    frame_indices: list[int],
    timestamps: list[float],
    player_positions: dict[int, list[Position]],
    player_speeds: dict[int, list[float | None]],
    player_distances: dict[int, list[float]],
    ball_positions: list[Position],
    ball_speeds: list[float | None],
    csv_path: str | Path,
) -> pd.DataFrame:
    """Write the full per-frame track log to CSV.

    Args:
        frame_indices: Source frame index for each processed frame.
        timestamps: Timestamp in seconds for each processed frame.
        player_positions: Court-space positions per player track ID.
        player_speeds: Speed series per player track ID.
        player_distances: Cumulative-distance series per player track ID.
        ball_positions: Court-space ball position per frame.
        ball_speeds: Ball speed series in km/h.
        csv_path: Destination CSV path.

    Returns:
        The written DataFrame, in long format with one row per frame per entity.
    """
    rows: list[dict[str, object]] = []
    for i, (frame_index, timestamp) in enumerate(zip(frame_indices, timestamps)):
        for player_id, positions in player_positions.items():
            position = positions[i]
            rows.append(
                {
                    "frame": frame_index,
                    "timestamp": round(timestamp, 3),
                    "entity": f"player_{player_id}",
                    "court_x_m": None if position is None else round(position[0], 3),
                    "court_y_m": None if position is None else round(position[1], 3),
                    "speed_kmh": _round_opt(player_speeds.get(player_id, [None] * len(timestamps))[i]),
                    "cumulative_distance_m": round(player_distances[player_id][i], 3),
                }
            )
        ball = ball_positions[i]
        rows.append(
            {
                "frame": frame_index,
                "timestamp": round(timestamp, 3),
                "entity": "ball",
                "court_x_m": None if ball is None else round(ball[0], 3),
                "court_y_m": None if ball is None else round(ball[1], 3),
                "speed_kmh": _round_opt(ball_speeds[i]),
                "cumulative_distance_m": None,
            }
        )

    frame = pd.DataFrame(rows)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _round_opt(value: float | None, digits: int = 2) -> float | None:
    """Round a value that may be ``None``."""
    return None if value is None else round(value, digits)
