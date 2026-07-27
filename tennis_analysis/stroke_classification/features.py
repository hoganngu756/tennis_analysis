"""Pose-sequence feature extraction for stroke classification.

Two representations are produced from the same pose window, so both classifier
backends share one feature definition:

* :func:`frame_features` — a ``(T, F)`` time series, consumed directly by the 1D-CNN.
* :func:`aggregate_features` — order-statistics over that series, giving the fixed
  length vector the RandomForest baseline needs.

Poses are normalised to be invariant to where the player is on court and how large
they appear: coordinates are re-centred on the hip midpoint and scaled by torso
length. Without this the classifier would key on the far player being smaller in frame
rather than on what their limbs are doing.
"""

from __future__ import annotations

import numpy as np

# --- COCO-17 keypoint indices, as emitted by YOLO11-pose ---------------------------
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

NUM_POSE_KEYPOINTS = 17

#: Names of the per-frame features, in the order :func:`frame_features` returns them.
FRAME_FEATURE_NAMES: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_wrist_y",
    "right_wrist_y",
    "left_wrist_x",
    "right_wrist_x",
    "wrist_separation",
    "torso_tilt",
    "highest_wrist_above_shoulder",
    "knee_flexion",
)

NUM_FRAME_FEATURES = len(FRAME_FEATURE_NAMES)

#: Aggregations applied to each per-frame feature.
_STAT_NAMES = ("mean", "std", "min", "max")
#: Aggregations applied to each per-frame feature's temporal derivative.
_VELOCITY_STAT_NAMES = ("vel_mean_abs", "vel_max_abs")

#: Names of the aggregate feature vector entries, in order.
AGGREGATE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{name}_{stat}" for name in FRAME_FEATURE_NAMES for stat in _STAT_NAMES
) + tuple(
    f"{name}_{stat}" for name in FRAME_FEATURE_NAMES for stat in _VELOCITY_STAT_NAMES
)

NUM_AGGREGATE_FEATURES = len(AGGREGATE_FEATURE_NAMES)


class PoseFeatureError(ValueError):
    """Raised when a pose window is malformed or too short to featurise."""


def normalize_pose_window(
    window: np.ndarray,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Centre and scale a pose window, blanking low-confidence keypoints.

    Args:
        window: ``(T, 17, 3)`` array of ``(x, y, confidence)`` per keypoint per frame.
            A ``(T, 17, 2)`` array is accepted and treated as fully confident.
        min_confidence: Keypoints scoring below this are replaced with NaN, so they
            propagate as "unknown" rather than as a spurious position at (0, 0).

    Returns:
        ``(T, 17, 2)`` normalised coordinates. The hip midpoint sits at the origin and
        distances are in units of torso length. The y axis is flipped so that positive
        means *upward*, matching physical intuition (image y grows downward).

    Raises:
        PoseFeatureError: If the input shape is not ``(T, 17, 2|3)`` or ``T`` is 0.
    """
    array = np.asarray(window, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != NUM_POSE_KEYPOINTS or array.shape[2] not in (2, 3):
        raise PoseFeatureError(
            f"expected pose window of shape (T, {NUM_POSE_KEYPOINTS}, 2 or 3), "
            f"got {array.shape}"
        )
    if array.shape[0] == 0:
        raise PoseFeatureError("pose window is empty")

    coords = array[:, :, :2].copy()
    if array.shape[2] == 3:
        coords[array[:, :, 2] < min_confidence] = np.nan

    hip_center = np.nanmean(coords[:, [LEFT_HIP, RIGHT_HIP], :], axis=1)
    shoulder_center = np.nanmean(coords[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :], axis=1)

    torso = np.linalg.norm(shoulder_center - hip_center, axis=1)
    # A collapsed or missing torso gives no usable scale; fall back to the median of
    # the frames that do have one, and finally to 1.0 (leaves coordinates in pixels).
    valid = np.isfinite(torso) & (torso > 1e-6)
    fallback = float(np.median(torso[valid])) if valid.any() else 1.0
    torso = np.where(valid, torso, fallback)

    centered = coords - hip_center[:, None, :]
    scaled = centered / torso[:, None, None]
    # Flip y so up is positive.
    scaled[:, :, 1] *= -1
    return scaled


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex ``b`` formed by the segments ``b->a`` and ``b->c``, in degrees.

    Args:
        a: ``(T, 2)`` positions of the first outer joint.
        b: ``(T, 2)`` positions of the vertex joint.
        c: ``(T, 2)`` positions of the second outer joint.

    Returns:
        ``(T,)`` angles in degrees in ``[0, 180]``. Frames with missing or coincident
        joints yield NaN.
    """
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)

    norm_ba = np.linalg.norm(ba, axis=-1)
    norm_bc = np.linalg.norm(bc, axis=-1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(ba * bc, axis=-1) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    degenerate = (norm_ba < 1e-9) | (norm_bc < 1e-9)
    return np.where(degenerate, np.nan, angles)


def frame_features(window: np.ndarray, min_confidence: float = 0.3) -> np.ndarray:
    """Compute the per-frame feature time series for a pose window.

    Args:
        window: ``(T, 17, 3)`` raw pose keypoints (or ``(T, 17, 2)``).
        min_confidence: Keypoint confidence floor, see :func:`normalize_pose_window`.

    Returns:
        ``(T, 12)`` array of per-frame features, matching :data:`FRAME_FEATURE_NAMES`.
        Missing values are NaN; use :func:`aggregate_features` or
        :func:`impute_frame_features` before feeding a model.

    Raises:
        PoseFeatureError: If the pose window is malformed.
    """
    pose = normalize_pose_window(window, min_confidence)

    left_elbow_angle = joint_angle(
        pose[:, LEFT_SHOULDER], pose[:, LEFT_ELBOW], pose[:, LEFT_WRIST]
    )
    right_elbow_angle = joint_angle(
        pose[:, RIGHT_SHOULDER], pose[:, RIGHT_ELBOW], pose[:, RIGHT_WRIST]
    )
    left_shoulder_angle = joint_angle(
        pose[:, LEFT_HIP], pose[:, LEFT_SHOULDER], pose[:, LEFT_ELBOW]
    )
    right_shoulder_angle = joint_angle(
        pose[:, RIGHT_HIP], pose[:, RIGHT_SHOULDER], pose[:, RIGHT_ELBOW]
    )

    left_wrist = pose[:, LEFT_WRIST]
    right_wrist = pose[:, RIGHT_WRIST]
    wrist_separation = np.linalg.norm(left_wrist - right_wrist, axis=1)

    # Torso tilt: signed angle of the shoulder line from horizontal. Distinguishes the
    # upright, rotated posture of a serve from the level shoulders of a groundstroke.
    shoulder_vector = pose[:, LEFT_SHOULDER] - pose[:, RIGHT_SHOULDER]
    torso_tilt = np.degrees(np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0]))

    shoulder_center_y = np.nanmean(
        pose[:, [LEFT_SHOULDER, RIGHT_SHOULDER], 1], axis=1
    )
    highest_wrist = np.nanmax(
        np.stack([left_wrist[:, 1], right_wrist[:, 1]], axis=1), axis=1
    )
    # Strongly positive on a serve (racket hand overhead), negative on groundstrokes.
    highest_wrist_above_shoulder = highest_wrist - shoulder_center_y

    left_knee_angle = joint_angle(pose[:, LEFT_HIP], pose[:, LEFT_KNEE], pose[:, LEFT_ANKLE])
    right_knee_angle = joint_angle(
        pose[:, RIGHT_HIP], pose[:, RIGHT_KNEE], pose[:, RIGHT_ANKLE]
    )
    knee_flexion = np.nanmean(np.stack([left_knee_angle, right_knee_angle], axis=1), axis=1)

    return np.stack(
        [
            left_elbow_angle,
            right_elbow_angle,
            left_shoulder_angle,
            right_shoulder_angle,
            left_wrist[:, 1],
            right_wrist[:, 1],
            left_wrist[:, 0],
            right_wrist[:, 0],
            wrist_separation,
            torso_tilt,
            highest_wrist_above_shoulder,
            knee_flexion,
        ],
        axis=1,
    )


def impute_frame_features(series: np.ndarray) -> np.ndarray:
    """Fill NaNs in a ``(T, F)`` feature series so a model can consume it.

    Each column is filled with its own finite mean; columns that are entirely NaN
    (e.g. a limb occluded for the whole window) become zeros.

    Args:
        series: ``(T, F)`` feature series, possibly containing NaN.

    Returns:
        A NaN-free ``(T, F)`` array.
    """
    filled = np.asarray(series, dtype=np.float64).copy()
    for column in range(filled.shape[1]):
        values = filled[:, column]
        finite = np.isfinite(values)
        values[~finite] = float(np.mean(values[finite])) if finite.any() else 0.0
    return filled


def aggregate_features(series: np.ndarray) -> np.ndarray:
    """Reduce a per-frame feature series to a fixed-length vector.

    Args:
        series: ``(T, 12)`` per-frame features from :func:`frame_features`.

    Returns:
        ``(72,)`` vector: mean/std/min/max of each feature, followed by the mean and
        max absolute temporal derivative of each. Always finite.

    Raises:
        PoseFeatureError: If ``series`` does not have 12 columns or has no frames.
    """
    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != NUM_FRAME_FEATURES:
        raise PoseFeatureError(
            f"expected series of shape (T, {NUM_FRAME_FEATURES}), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise PoseFeatureError("feature series is empty")

    filled = impute_frame_features(array)

    stats = np.stack(
        [filled.mean(axis=0), filled.std(axis=0), filled.min(axis=0), filled.max(axis=0)],
        axis=1,
    )  # (F, 4)

    if filled.shape[0] > 1:
        velocity = np.diff(filled, axis=0)
    else:
        # A single-frame window has no motion; report zero rather than failing.
        velocity = np.zeros((1, filled.shape[1]))
    velocity_stats = np.stack(
        [np.abs(velocity).mean(axis=0), np.abs(velocity).max(axis=0)], axis=1
    )  # (F, 2)

    return np.concatenate([stats.reshape(-1), velocity_stats.reshape(-1)])


def extract_features(window: np.ndarray, min_confidence: float = 0.3) -> np.ndarray:
    """Convenience wrapper: raw pose window straight to an aggregate feature vector.

    Args:
        window: ``(T, 17, 3)`` raw pose keypoints.
        min_confidence: Keypoint confidence floor.

    Returns:
        ``(72,)`` aggregate feature vector.
    """
    return aggregate_features(frame_features(window, min_confidence))


def resample_series(series: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a ``(T, F)`` series to a fixed number of timesteps.

    The 1D-CNN backend needs uniform-length inputs, but shot windows vary with FPS and
    with clipping at video boundaries.

    Args:
        series: ``(T, F)`` feature series.
        length: Target number of timesteps. Must be >= 1.

    Returns:
        ``(length, F)`` resampled series.

    Raises:
        PoseFeatureError: If ``series`` is empty or ``length`` < 1.
    """
    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise PoseFeatureError(f"expected non-empty (T, F) series, got {array.shape}")
    if length < 1:
        raise PoseFeatureError(f"length must be >= 1, got {length}")

    if array.shape[0] == 1:
        return np.repeat(array, length, axis=0)

    source = np.linspace(0.0, 1.0, array.shape[0])
    target = np.linspace(0.0, 1.0, length)
    return np.stack(
        [np.interp(target, source, array[:, c]) for c in range(array.shape[1])], axis=1
    )
