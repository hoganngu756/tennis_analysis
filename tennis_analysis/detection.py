"""Stages 2 & 3 — player detection/tracking and ball detection.

Players use a pretrained YOLO11 model restricted to the COCO ``person`` class, with
Ultralytics' built-in ByteTrack for stable IDs across frames. The ball uses a separate
YOLO11 model fine-tuned on tennis-ball data (see ``scripts/train_ball_detector.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import BallConfig, PlayerConfig
from .types import BBox, PlayerDetection

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

#: COCO class index for "person".
PERSON_CLASS_ID = 0


def _load_yolo(model_path: str, purpose: str) -> "YOLO":
    """Load an Ultralytics YOLO model, with a clear error when weights are missing.

    Args:
        model_path: Path to a ``.pt`` checkpoint, or a bare model name that Ultralytics
            will download (e.g. ``yolo11n.pt``).
        purpose: Human-readable description used in error messages.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If ``model_path`` looks like a local path but is absent.
    """
    from ultralytics import YOLO

    path = Path(model_path)
    # A bare name like "yolo11n.pt" has no directory component and is auto-downloaded;
    # anything with a directory is expected to exist locally.
    if path.parent != Path(".") and not path.exists():
        raise FileNotFoundError(
            f"{purpose} weights not found at {model_path}. "
            f"Train them first (see README) or update config.yaml."
        )
    return YOLO(model_path)


class PlayerTracker:
    """Detects and tracks people with YOLO11 + ByteTrack.

    Tracking state persists across calls, so frames must be fed in chronological order.
    """

    def __init__(self, config: PlayerConfig) -> None:
        """Load the detection model.

        Args:
            config: Player detection settings.
        """
        self.config = config
        self._model = _load_yolo(config.model, "player detector")

    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        """Detect and track people in one frame.

        Detections without a track ID (ByteTrack needs a few frames to confirm a new
        track) are dropped, since downstream stages key everything on track ID.

        Args:
            frame: BGR image.

        Returns:
            One :class:`PlayerDetection` per confirmed track, with ``court_xy`` unset.
        """
        results = self._model.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            classes=[PERSON_CLASS_ID],
            conf=self.config.confidence,
            verbose=False,
        )
        return _parse_tracked_boxes(results)

    def reset(self) -> None:
        """Clear tracker state so a new video starts with fresh track IDs."""
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            trackers = getattr(self._model.predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    tracker.reset()


def _parse_tracked_boxes(results: Any) -> list[PlayerDetection]:
    """Convert Ultralytics tracking results into :class:`PlayerDetection` objects."""
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    return [
        PlayerDetection(
            track_id=int(track_id),
            bbox=BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            confidence=float(conf),
        )
        for box, track_id, conf in zip(xyxy, ids, confs)
    ]


class BallDetector:
    """Detects the tennis ball with a fine-tuned YOLO11 model.

    Only the single highest-confidence box is kept per frame — there is exactly one
    ball in play, and extra boxes are almost always false positives on line markings
    or background clutter.
    """

    def __init__(self, config: BallConfig) -> None:
        """Load the ball detection model.

        Args:
            config: Ball detection settings.

        Raises:
            FileNotFoundError: If the fine-tuned weights are missing.
        """
        self.config = config
        self._model = _load_yolo(config.model, "ball detector")

    def detect(self, frame: np.ndarray) -> tuple[tuple[float, float] | None, float]:
        """Detect the ball in one frame.

        Args:
            frame: BGR image.

        Returns:
            ``(center_xy, confidence)``. ``center_xy`` is ``None`` when nothing was
            detected above the confidence threshold, in which case confidence is 0.0.
        """
        results = self._model.predict(
            frame,
            conf=self.config.confidence,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        if not results:
            return None, 0.0

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0

        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[best]
        return ((float(x1 + x2) / 2, float(y1 + y2) / 2), float(confs[best]))
