"""Stage 6 — pose-based stroke classification."""

from .classifier import (
    CNN1DStrokeClassifier,
    RandomForestStrokeClassifier,
    StrokeClassifier,
    build_classifier,
    load_classifier,
)
from .features import (
    AGGREGATE_FEATURE_NAMES,
    FRAME_FEATURE_NAMES,
    NUM_AGGREGATE_FEATURES,
    NUM_FRAME_FEATURES,
    PoseFeatureError,
    aggregate_features,
    extract_features,
    frame_features,
    joint_angle,
    normalize_pose_window,
    resample_series,
)
from .pose import PoseEstimator, build_pose_window
from .shot_detection import ShotMoment, assign_striker, detect_shot_moments, window_bounds

__all__ = [
    "StrokeClassifier",
    "RandomForestStrokeClassifier",
    "CNN1DStrokeClassifier",
    "build_classifier",
    "load_classifier",
    "PoseEstimator",
    "build_pose_window",
    "ShotMoment",
    "detect_shot_moments",
    "assign_striker",
    "window_bounds",
    "PoseFeatureError",
    "normalize_pose_window",
    "joint_angle",
    "frame_features",
    "aggregate_features",
    "extract_features",
    "resample_series",
    "FRAME_FEATURE_NAMES",
    "AGGREGATE_FEATURE_NAMES",
    "NUM_FRAME_FEATURES",
    "NUM_AGGREGATE_FEATURES",
]
