"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        """Box centre ``(x, y)``."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def feet(self) -> tuple[float, float]:
        """Bottom-centre of the box — the player's approximate ground contact point.

        This is what gets projected through the homography: the box centre floats at
        chest height and would land metres deep into the court.
        """
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as a plain ``(x1, y1, x2, y2)`` tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def clip(self, width: int, height: int) -> "BBox":
        """Clamp the box to image bounds.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            A new box guaranteed to lie inside ``[0, width] x [0, height]``.
        """
        return BBox(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
        )


@dataclass
class PlayerDetection:
    """One tracked player in one frame."""

    track_id: int
    bbox: BBox
    confidence: float
    #: Court-space position in metres, or ``None`` when no homography was available.
    court_xy: tuple[float, float] | None = None
    #: Pose keypoints as ``(17, 3)`` — x, y (pixels) and confidence — if pose ran.
    pose: np.ndarray | None = None


@dataclass
class BallDetection:
    """The ball in one frame, after Kalman smoothing."""

    #: Smoothed pixel position. Always populated while the track is alive.
    position: tuple[float, float]
    confidence: float
    #: True when this frame had no raw detection and the position is a prediction.
    interpolated: bool = False
    court_xy: tuple[float, float] | None = None


@dataclass
class FrameResult:
    """Everything the pipeline knows about a single processed frame."""

    frame_index: int
    timestamp: float
    players: list[PlayerDetection] = field(default_factory=list)
    ball: BallDetection | None = None
    #: Detected court keypoints as ``(14, 2)`` pixels, if the court model ran.
    court_keypoints: np.ndarray | None = None
    #: Per-player speed in km/h, keyed by track id.
    player_speeds: dict[int, float] = field(default_factory=dict)
    #: Per-player cumulative distance in metres, keyed by track id.
    player_distances: dict[int, float] = field(default_factory=dict)
    ball_speed_kmh: float | None = None


@dataclass
class Shot:
    """A detected and classified stroke."""

    frame_index: int
    timestamp: float
    player_id: int
    stroke: str
    stroke_confidence: float
    ball_speed_kmh: float | None = None
    player_speed_kmh: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Flat mapping suitable for CSV/JSON export."""
        return {
            "frame": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "player": self.player_id,
            "stroke": self.stroke,
            "stroke_confidence": round(self.stroke_confidence, 4),
            "ball_speed_kmh": (
                None if self.ball_speed_kmh is None else round(self.ball_speed_kmh, 2)
            ),
            "player_speed_kmh": (
                None if self.player_speed_kmh is None else round(self.player_speed_kmh, 2)
            ),
        }
