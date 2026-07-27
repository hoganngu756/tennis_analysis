"""Stage 4 — court keypoint detection, homography and the mini-court view."""

from .detector import CourtKeypointDetector, HomographyTracker, select_device
from .geometry import (
    COURT_KEYPOINTS_M,
    COURT_LENGTH,
    DOUBLES_WIDTH,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SINGLES_WIDTH,
    CourtHomography,
    HomographyError,
    is_inside_court,
)
from .mini_court import MiniCourt
from .model import CourtKeypointNet, keypoint_loss

__all__ = [
    "CourtKeypointDetector",
    "HomographyTracker",
    "select_device",
    "CourtHomography",
    "HomographyError",
    "COURT_KEYPOINTS_M",
    "KEYPOINT_NAMES",
    "NUM_KEYPOINTS",
    "COURT_LENGTH",
    "DOUBLES_WIDTH",
    "SINGLES_WIDTH",
    "is_inside_court",
    "MiniCourt",
    "CourtKeypointNet",
    "keypoint_loss",
]
