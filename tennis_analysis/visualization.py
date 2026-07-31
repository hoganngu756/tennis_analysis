"""Stage 7 — drawing detections, speeds, stroke labels and the mini-court."""

from __future__ import annotations

import cv2
import numpy as np

from .court_keypoints.mini_court import PLAYER_COLORS, MiniCourt
from .types import BallDetection, PlayerDetection

_BALL_COLOR = (0, 255, 255)
_KEYPOINT_COLOR = (0, 200, 0)
_TEXT_COLOR = (255, 255, 255)
_STROKE_BG = (30, 30, 200)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for_player(player_id: int, ordered_ids: list[int]) -> tuple[int, int, int]:
    """Pick a stable colour for a player track.

    Args:
        player_id: The player's track ID.
        ordered_ids: All player IDs, in a stable order (e.g. sorted).

    Returns:
        A BGR colour tuple.
    """
    try:
        slot = ordered_ids.index(player_id)
    except ValueError:
        slot = player_id
    return PLAYER_COLORS[slot % len(PLAYER_COLORS)]


def draw_players(
    frame: np.ndarray,
    players: list[PlayerDetection],
    ordered_ids: list[int],
    speeds: dict[int, float] | None = None,
    distances: dict[int, float] | None = None,
) -> np.ndarray:
    """Draw player boxes, IDs and optional speed/distance readouts.

    Args:
        frame: BGR frame, modified in place.
        players: Players to draw.
        ordered_ids: Stable player ordering, used for colour assignment.
        speeds: Current speed in km/h per track ID.
        distances: Cumulative distance in metres per track ID.

    Returns:
        The same ``frame``, for chaining.
    """
    for player in players:
        color = color_for_player(player.track_id, ordered_ids)
        x1, y1, x2, y2 = (int(round(v)) for v in player.bbox.to_xyxy())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        lines = [f"Player {player.track_id}"]
        if speeds is not None and player.track_id in speeds:
            lines.append(f"{speeds[player.track_id]:.1f} km/h")
        if distances is not None and player.track_id in distances:
            lines.append(f"{distances[player.track_id]:.1f} m")

        for offset, text in enumerate(lines):
            _draw_label(frame, text, (x1, y1 - 8 - offset * 18), color)
    return frame


def draw_ball(
    frame: np.ndarray, ball: BallDetection | None, speed_kmh: float | None = None
) -> np.ndarray:
    """Draw the ball marker and optional speed readout.

    Interpolated positions are drawn hollow so it is visually obvious when the ball
    is being predicted through an occlusion rather than actually detected.

    Args:
        frame: BGR frame, modified in place.
        ball: The ball detection, or ``None``.
        speed_kmh: Ball speed to annotate, if known.

    Returns:
        The same ``frame``, for chaining.
    """
    if ball is None:
        return frame

    x, y = (int(round(v)) for v in ball.position)
    if ball.interpolated:
        cv2.circle(frame, (x, y), 7, _BALL_COLOR, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), 6, _BALL_COLOR, -1, cv2.LINE_AA)

    if speed_kmh is not None:
        _draw_label(frame, f"Ball {speed_kmh:.0f} km/h", (x + 12, y - 8), _BALL_COLOR)
    return frame


def draw_court_keypoints(frame: np.ndarray, keypoints: np.ndarray | None) -> np.ndarray:
    """Draw detected court keypoints as small numbered dots.

    Args:
        frame: BGR frame, modified in place.
        keypoints: ``(14, 2)`` pixel coordinates, or ``None``.

    Returns:
        The same ``frame``, for chaining.
    """
    if keypoints is None:
        return frame
    for index, point in enumerate(keypoints):
        if not np.isfinite(point).all():
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(frame, (x, y), 3, _KEYPOINT_COLOR, -1, cv2.LINE_AA)
        cv2.putText(frame, str(index), (x + 4, y - 4), _FONT, 0.35, _KEYPOINT_COLOR, 1)
    return frame


def draw_stroke_label(
    frame: np.ndarray, bbox_top_left: tuple[int, int], stroke: str, confidence: float
) -> np.ndarray:
    """Draw a stroke-type callout above a player.

    Args:
        frame: BGR frame, modified in place.
        bbox_top_left: ``(x, y)`` anchor, typically the player's box corner.
        stroke: Predicted stroke name.
        confidence: Prediction confidence in ``[0, 1]``.

    Returns:
        The same ``frame``, for chaining.
    """
    # A shot moment found without a classifier has no meaningful confidence; showing
    # "0%" next to it reads as a failed prediction rather than an unlabelled one.
    text = f"{stroke.upper()} {confidence:.0%}" if confidence > 0 else stroke.upper()
    x, y = bbox_top_left
    _draw_label(frame, text, (x, max(24, y - 46)), _TEXT_COLOR, background=_STROKE_BG, scale=0.6)
    return frame


def draw_frame_info(frame: np.ndarray, frame_index: int, timestamp: float) -> np.ndarray:
    """Draw the frame number and timestamp in the top-left corner.

    Args:
        frame: BGR frame, modified in place.
        frame_index: Source frame number.
        timestamp: Timestamp in seconds.

    Returns:
        The same ``frame``, for chaining.
    """
    _draw_label(frame, f"frame {frame_index}  t={timestamp:.2f}s", (12, 24), _TEXT_COLOR)
    return frame


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    background: tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.5,
) -> None:
    """Draw text on a filled background box so it stays legible over any footage."""
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = origin
    x = max(0, min(x, frame.shape[1] - width - 4))
    y = max(height + 4, min(y, frame.shape[0] - 4))

    cv2.rectangle(
        frame,
        (x - 3, y - height - baseline),
        (x + width + 3, y + baseline - 1),
        background,
        -1,
    )
    cv2.putText(frame, text, (x, y - baseline + 1), _FONT, scale, color, thickness, cv2.LINE_AA)


def render_mini_court(
    frame: np.ndarray,
    mini_court: MiniCourt,
    player_court_positions: dict[int, tuple[float, float]],
    ball_court_position: tuple[float, float] | None,
) -> np.ndarray:
    """Render and composite the top-down mini-court onto a frame.

    Args:
        frame: BGR frame, modified in place.
        mini_court: The configured mini-court renderer.
        player_court_positions: Court-space player positions in metres, by track ID.
        ball_court_position: Court-space ball position in metres, if known.

    Returns:
        The same ``frame``, for chaining.
    """
    canvas = mini_court.render(player_court_positions, ball_court_position)
    return mini_court.overlay(frame, canvas)
