"""2D top-down mini-court rendering.

Court-space metres are mapped to a small pixel canvas that gets composited onto the
annotated video, giving a bird's-eye view of where both players and the ball are.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import MiniCourtConfig
from .geometry import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    NET_Y,
    SERVICE_LINE_FAR_Y,
    SERVICE_LINE_NEAR_Y,
    SINGLES_INSET,
)

#: BGR colours for the mini-court render.
_SURFACE_COLOR = (90, 60, 30)
_LINE_COLOR = (240, 240, 240)
_NET_COLOR = (200, 200, 200)
_BALL_COLOR = (0, 255, 255)
#: Distinct colours cycled per player track.
PLAYER_COLORS: tuple[tuple[int, int, int], ...] = ((0, 140, 255), (255, 140, 0))


class MiniCourt:
    """Renders a top-down court view and composites it onto video frames."""

    def __init__(self, config: MiniCourtConfig) -> None:
        """Prepare the mini-court canvas.

        Args:
            config: Overlay size, margin and corner placement.
        """
        self.config = config
        # Inner padding so player markers near the baseline are not clipped.
        self._pad = 12
        self._draw_w = config.width - 2 * self._pad
        self._draw_h = config.height - 2 * self._pad
        self._base = self._render_court()

    def court_to_canvas(self, court_xy: tuple[float, float]) -> tuple[int, int]:
        """Map a court-space point in metres to mini-court canvas pixels.

        Args:
            court_xy: ``(x, y)`` in metres in the court frame.

        Returns:
            ``(px, py)`` integer pixel coordinates on the mini-court canvas.
        """
        x, y = court_xy
        px = self._pad + (x / DOUBLES_WIDTH) * self._draw_w
        py = self._pad + (y / COURT_LENGTH) * self._draw_h
        return int(round(px)), int(round(py))

    def _render_court(self) -> np.ndarray:
        """Draw the static court lines once, to be copied per frame."""
        canvas = np.full(
            (self.config.height, self.config.width, 3), _SURFACE_COLOR, dtype=np.uint8
        )

        def line(p1: tuple[float, float], p2: tuple[float, float], color, thickness=1):
            cv2.line(
                canvas,
                self.court_to_canvas(p1),
                self.court_to_canvas(p2),
                color,
                thickness,
                cv2.LINE_AA,
            )

        singles_left = SINGLES_INSET
        singles_right = DOUBLES_WIDTH - SINGLES_INSET
        center_x = DOUBLES_WIDTH / 2

        # Doubles boundary.
        cv2.rectangle(
            canvas,
            self.court_to_canvas((0, 0)),
            self.court_to_canvas((DOUBLES_WIDTH, COURT_LENGTH)),
            _LINE_COLOR,
            2,
        )
        # Singles sidelines.
        line((singles_left, 0), (singles_left, COURT_LENGTH), _LINE_COLOR)
        line((singles_right, 0), (singles_right, COURT_LENGTH), _LINE_COLOR)
        # Service lines.
        line((singles_left, SERVICE_LINE_NEAR_Y), (singles_right, SERVICE_LINE_NEAR_Y), _LINE_COLOR)
        line((singles_left, SERVICE_LINE_FAR_Y), (singles_right, SERVICE_LINE_FAR_Y), _LINE_COLOR)
        # Centre service line.
        line((center_x, SERVICE_LINE_NEAR_Y), (center_x, SERVICE_LINE_FAR_Y), _LINE_COLOR)
        # Net.
        line((0, NET_Y), (DOUBLES_WIDTH, NET_Y), _NET_COLOR, 2)

        return canvas

    def render(
        self,
        players: dict[int, tuple[float, float]],
        ball: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Render one mini-court frame.

        Args:
            players: Court-space positions in metres, keyed by player track ID.
            ball: Court-space ball position in metres, if known.

        Returns:
            The rendered BGR canvas.
        """
        canvas = self._base.copy()

        for slot, (track_id, position) in enumerate(sorted(players.items())):
            if not np.isfinite(position).all():
                continue
            color = PLAYER_COLORS[slot % len(PLAYER_COLORS)]
            point = self.court_to_canvas(position)
            cv2.circle(canvas, point, 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(track_id),
                (point[0] + 7, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        if ball is not None and np.isfinite(ball).all():
            cv2.circle(canvas, self.court_to_canvas(ball), 3, _BALL_COLOR, -1, cv2.LINE_AA)

        return canvas

    def overlay(self, frame: np.ndarray, mini: np.ndarray, alpha: float = 0.85) -> np.ndarray:
        """Composite a rendered mini-court onto a video frame in place.

        Args:
            frame: BGR video frame, modified in place.
            mini: Rendered mini-court canvas.
            alpha: Opacity of the overlay.

        Returns:
            The same ``frame`` object, for chaining.
        """
        height, width = frame.shape[:2]
        margin = self.config.margin
        mh, mw = mini.shape[:2]

        if mh + 2 * margin > height or mw + 2 * margin > width:
            # Overlay does not fit; skip rather than crash on a small input video.
            return frame

        top = margin if self.config.position.startswith("top") else height - mh - margin
        left = margin if self.config.position.endswith("left") else width - mw - margin

        region = frame[top : top + mh, left : left + mw]
        cv2.addWeighted(mini, alpha, region, 1 - alpha, 0, region)
        return frame
