"""Pose estimation over tracked player crops.

YOLO11-pose is run on the player's bounding-box crop rather than the whole frame. On
broadcast footage a player occupies a small fraction of the image; cropping first
raises the effective resolution on the limbs that carry the stroke signal, and it
sidesteps having to match whole-frame pose detections back to track IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..types import BBox
from .features import NUM_POSE_KEYPOINTS

logger = logging.getLogger(__name__)

#: Fractional padding added around the player box before cropping. A tight box clips
#: the racket arm at full extension, which is precisely the frame that matters.
CROP_PADDING = 0.15


class PoseEstimator:
    """Estimates 17-keypoint COCO poses for tracked players."""

    def __init__(self, model_path: str = "yolo11n-pose.pt", confidence: float = 0.3) -> None:
        """Load the pose model.

        Args:
            model_path: Ultralytics pose checkpoint, or a bare name to auto-download.
            confidence: Detection confidence threshold.

        Raises:
            FileNotFoundError: If a local checkpoint path is given but does not exist.
        """
        from ultralytics import YOLO

        path = Path(model_path)
        if path.parent != Path(".") and not path.exists():
            raise FileNotFoundError(f"pose model not found at {model_path}")

        self._model = YOLO(model_path)
        self.confidence = confidence

    def estimate(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        """Estimate the pose of the player inside ``bbox``.

        Args:
            frame: Full BGR video frame.
            bbox: The player's bounding box in frame coordinates.

        Returns:
            ``(17, 3)`` array of ``(x, y, confidence)`` with x/y in *full-frame* pixel
            coordinates, or ``None`` if no pose was found.
        """
        height, width = frame.shape[:2]
        crop_box = _pad_box(bbox, width, height)
        if crop_box.width < 2 or crop_box.height < 2:
            return None

        x1, y1, x2, y2 = (int(round(v)) for v in crop_box.to_xyxy())
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        results = self._model.predict(crop, conf=self.confidence, verbose=False)
        if not results:
            return None

        keypoints = results[0].keypoints
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return None

        # Multiple people can fall inside a padded box (a nearby opponent, a line
        # judge); keep the one with the most confident keypoints overall.
        data = keypoints.data.cpu().numpy()  # (N, 17, 3)
        best = int(np.argmax(data[:, :, 2].mean(axis=1)))
        pose = data[best].astype(np.float64)

        if pose.shape != (NUM_POSE_KEYPOINTS, 3):
            logger.debug("unexpected pose shape %s, skipping", pose.shape)
            return None

        # Crop-local -> full-frame coordinates.
        pose[:, 0] += x1
        pose[:, 1] += y1
        return pose


def _pad_box(bbox: BBox, width: int, height: int, padding: float = CROP_PADDING) -> BBox:
    """Expand a box by a fractional padding, clipped to the frame."""
    pad_x = bbox.width * padding
    pad_y = bbox.height * padding
    return BBox(
        bbox.x1 - pad_x, bbox.y1 - pad_y, bbox.x2 + pad_x, bbox.y2 + pad_y
    ).clip(width, height)


def build_pose_window(
    poses: list[np.ndarray | None], start: int, end: int
) -> np.ndarray | None:
    """Assemble a contiguous pose window, tolerating frames where pose failed.

    Args:
        poses: Per-frame poses for one player, ``None`` where estimation failed.
        start: Inclusive start index.
        end: Exclusive end index.

    Returns:
        ``(T, 17, 3)`` array with missing frames filled by zero-confidence keypoints
        (which the feature extractor blanks to NaN), or ``None`` if every frame in the
        window is missing.
    """
    segment = poses[start:end]
    if not segment or all(pose is None for pose in segment):
        return None

    blank = np.zeros((NUM_POSE_KEYPOINTS, 3), dtype=np.float64)
    return np.stack([blank if pose is None else pose for pose in segment])
