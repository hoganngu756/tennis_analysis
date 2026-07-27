"""Shot-moment detection from the ball trajectory.

A stroke is the instant the racket reverses the ball. In trajectory terms that is a
sharp change in direction, a sharp change in speed, or both — which is exactly what
this module looks for. Working from the Kalman-smoothed track keeps the derivative
estimates stable enough for the angle test to mean something.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Position = tuple[float, float] | None


@dataclass(frozen=True)
class ShotMoment:
    """A candidate instant at which a player struck the ball."""

    #: Index into the processed-frame sequence (not the source frame number).
    index: int
    #: Direction change in degrees at this index.
    direction_change_deg: float
    #: Absolute change in speed magnitude, in the units of the input positions.
    speed_change: float
    #: Combined salience, used to rank competing candidates during suppression.
    score: float


def _velocities(positions: list[Position], smoothing: int = 2) -> np.ndarray:
    """Estimate per-frame velocity via a centred difference over ``smoothing`` frames.

    Returns an ``(N, 2)`` array with NaN where velocity cannot be computed.
    """
    count = len(positions)
    array = np.full((count, 2), np.nan)
    for i, position in enumerate(positions):
        if position is not None:
            array[i] = position

    velocity = np.full((count, 2), np.nan)
    for i in range(count):
        before = max(0, i - smoothing)
        after = min(count - 1, i + smoothing)
        if after == before:
            continue
        start, end = array[before], array[after]
        if np.isfinite(start).all() and np.isfinite(end).all():
            velocity[i] = (end - start) / (after - before)
    return velocity


def detect_shot_moments(
    ball_positions: list[Position],
    fps: float,
    min_direction_change_deg: float = 45.0,
    min_speed_change: float = 4.0,
    min_shot_interval_s: float = 0.4,
) -> list[ShotMoment]:
    """Find candidate shot moments in a ball trajectory.

    Args:
        ball_positions: Smoothed ball position per processed frame, ``None`` where the
            track was dead. Pixel or court coordinates both work; thresholds must
            match the chosen units.
        fps: Frame rate of the processed-frame sequence.
        min_direction_change_deg: Minimum direction reversal to qualify as a strike.
        min_speed_change: Minimum absolute speed change to qualify as a strike.
        min_shot_interval_s: Candidates closer together than this are suppressed,
            keeping only the highest-scoring one. Prevents a single impact from
            registering as several shots across adjacent frames.

    Returns:
        Shot moments in chronological order.

    Raises:
        ValueError: If ``fps`` <= 0.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if len(ball_positions) < 3:
        return []

    velocity = _velocities(ball_positions)
    speed = np.linalg.norm(velocity, axis=1)

    candidates: list[ShotMoment] = []
    for i in range(1, len(ball_positions) - 1):
        before, after = velocity[i - 1], velocity[i + 1]
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            continue
        if np.linalg.norm(before) < 1e-6 or np.linalg.norm(after) < 1e-6:
            continue

        cosine = np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
        direction_change = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        speed_change = float(abs(speed[i + 1] - speed[i - 1]))

        if direction_change < min_direction_change_deg and speed_change < min_speed_change:
            continue

        # Direction reversal is the primary cue and is already bounded at 1.0 by the
        # 180-degree normalisation. The speed term is squashed to at most 0.5 so a
        # large raw speed change cannot outvote a genuine reversal — without the
        # squash, the peak lands a frame or two *before* impact, where the ball is
        # decelerating but has not yet turned around, and the pose window centres on
        # the wrong instant.
        score = direction_change / 180.0 + 0.5 * float(
            np.tanh(speed_change / max(min_speed_change, 1e-6))
        )
        candidates.append(
            ShotMoment(
                index=i,
                direction_change_deg=direction_change,
                speed_change=speed_change,
                score=score,
            )
        )

    return _suppress_neighbours(candidates, int(round(min_shot_interval_s * fps)))


def _suppress_neighbours(
    candidates: list[ShotMoment], min_gap_frames: int
) -> list[ShotMoment]:
    """Greedy non-maximum suppression over shot candidates by frame distance."""
    if not candidates:
        return []
    if min_gap_frames <= 0:
        return sorted(candidates, key=lambda c: c.index)

    kept: list[ShotMoment] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if all(abs(candidate.index - other.index) >= min_gap_frames for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda c: c.index)


def assign_striker(
    ball_position: Position,
    player_positions: dict[int, Position],
    max_distance: float | None = None,
) -> int | None:
    """Attribute a shot to the player closest to the ball at that instant.

    Args:
        ball_position: Ball position at the shot moment.
        player_positions: Player positions keyed by track ID, in the same coordinate
            space as ``ball_position``.
        max_distance: If set, reject the assignment when the nearest player is further
            away than this — a shot with no plausible striker is better left
            unattributed than pinned on the wrong player.

    Returns:
        The striking player's track ID, or ``None`` if no player qualifies.
    """
    if ball_position is None or not np.isfinite(ball_position).all():
        return None

    best_id: int | None = None
    best_distance = np.inf
    for player_id, position in player_positions.items():
        if position is None or not np.isfinite(position).all():
            continue
        distance = float(np.hypot(position[0] - ball_position[0], position[1] - ball_position[1]))
        if distance < best_distance:
            best_distance, best_id = distance, player_id

    if best_id is None:
        return None
    if max_distance is not None and best_distance > max_distance:
        return None
    return best_id


def window_bounds(
    index: int, window_seconds: float, fps: float, total_frames: int
) -> tuple[int, int]:
    """Compute the half-open frame range of the pose window around a shot moment.

    The range is clipped to the video, so shots near the start or end yield a shorter
    window rather than an out-of-bounds slice.

    Args:
        index: Frame index of the shot moment.
        window_seconds: Half-width of the window, in seconds.
        fps: Frame rate of the processed-frame sequence.
        total_frames: Total number of processed frames.

    Returns:
        ``(start, end)`` with ``end`` exclusive.

    Raises:
        ValueError: If ``fps`` <= 0.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    half = max(1, int(round(window_seconds * fps)))
    return max(0, index - half), min(total_frames, index + half + 1)
