"""Tests for stroke feature extraction and shot-moment detection.

Poses are synthesised so the expected geometry is known: a straight arm must measure
180 degrees, a raised arm must read as above the shoulder, and a player scaled up or
moved across court must produce identical normalised features.
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis_analysis.stroke_classification.features import (
    AGGREGATE_FEATURE_NAMES,
    FRAME_FEATURE_NAMES,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NUM_AGGREGATE_FEATURES,
    NUM_FRAME_FEATURES,
    NUM_POSE_KEYPOINTS,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseFeatureError,
    aggregate_features,
    extract_features,
    frame_features,
    impute_frame_features,
    joint_angle,
    normalize_pose_window,
    resample_series,
)
from tennis_analysis.stroke_classification.shot_detection import (
    assign_striker,
    detect_shot_moments,
    window_bounds,
)


def make_pose(
    *,
    right_wrist: tuple[float, float] = (140.0, 200.0),
    left_wrist: tuple[float, float] = (60.0, 200.0),
    right_elbow: tuple[float, float] = (130.0, 170.0),
    left_elbow: tuple[float, float] = (70.0, 170.0),
    confidence: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    """Build a single plausible standing pose as a ``(17, 3)`` array.

    Image coordinates, so smaller y means higher up the frame. The default layout is
    an upright player with both arms down and slightly out.
    """
    pose = np.zeros((NUM_POSE_KEYPOINTS, 3), dtype=np.float64)
    layout = {
        0: (100.0, 100.0),  # nose
        LEFT_SHOULDER: (80.0, 140.0),
        RIGHT_SHOULDER: (120.0, 140.0),
        LEFT_ELBOW: left_elbow,
        RIGHT_ELBOW: right_elbow,
        LEFT_WRIST: left_wrist,
        RIGHT_WRIST: right_wrist,
        LEFT_HIP: (85.0, 220.0),
        RIGHT_HIP: (115.0, 220.0),
        LEFT_KNEE: (85.0, 290.0),
        14: (115.0, 290.0),  # right knee
        15: (85.0, 360.0),  # left ankle
        16: (115.0, 360.0),  # right ankle
    }
    for index in range(NUM_POSE_KEYPOINTS):
        x, y = layout.get(index, (100.0, 150.0))
        pose[index] = (origin[0] + x * scale, origin[1] + y * scale, confidence)
    return pose


def make_window(count: int = 8, **kwargs) -> np.ndarray:
    """Stack ``count`` identical poses into a ``(T, 17, 3)`` window."""
    return np.stack([make_pose(**kwargs) for _ in range(count)])


class TestJointAngle:
    """Angle at a vertex joint, in degrees."""

    def test_right_angle(self):
        a = np.array([[0.0, 1.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[1.0, 0.0]])
        assert joint_angle(a, b, c)[0] == pytest.approx(90.0)

    def test_straight_limb_is_180_degrees(self):
        a = np.array([[-1.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[1.0, 0.0]])
        assert joint_angle(a, b, c)[0] == pytest.approx(180.0)

    def test_fully_folded_limb_is_zero(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[2.0, 0.0]])
        assert joint_angle(a, b, c)[0] == pytest.approx(0.0)

    def test_45_degrees(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[1.0, 1.0]])
        assert joint_angle(a, b, c)[0] == pytest.approx(45.0)

    def test_coincident_joints_give_nan(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[1.0, 0.0]])
        assert np.isnan(joint_angle(a, b, c)[0])

    def test_nan_input_propagates(self):
        a = np.array([[np.nan, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[1.0, 0.0]])
        assert np.isnan(joint_angle(a, b, c)[0])

    def test_vectorised_over_time(self):
        a = np.array([[0.0, 1.0], [-1.0, 0.0]])
        b = np.zeros((2, 2))
        c = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(joint_angle(a, b, c), [90.0, 180.0])

    def test_always_within_zero_to_180(self):
        rng = np.random.default_rng(3)
        a, b, c = (rng.normal(0, 10, (200, 2)) for _ in range(3))
        angles = joint_angle(a, b, c)
        finite = angles[np.isfinite(angles)]
        assert ((finite >= 0.0) & (finite <= 180.0)).all()


class TestNormalizePoseWindow:
    """Translation, scale and confidence handling."""

    def test_output_shape_drops_confidence_channel(self):
        assert normalize_pose_window(make_window(6)).shape == (6, NUM_POSE_KEYPOINTS, 2)

    def test_hip_midpoint_is_at_the_origin(self):
        pose = normalize_pose_window(make_window(4))
        hip_center = pose[:, [LEFT_HIP, RIGHT_HIP], :].mean(axis=1)
        np.testing.assert_allclose(hip_center, 0.0, atol=1e-9)

    def test_invariant_to_court_position(self):
        near = normalize_pose_window(make_window(4, origin=(0.0, 0.0)))
        far = normalize_pose_window(make_window(4, origin=(900.0, 40.0)))
        np.testing.assert_allclose(near, far, atol=1e-9)

    def test_invariant_to_apparent_size(self):
        """The far player is smaller in frame but must featurise identically."""
        big = normalize_pose_window(make_window(4, scale=1.0))
        small = normalize_pose_window(make_window(4, scale=0.35))
        np.testing.assert_allclose(big, small, atol=1e-9)

    def test_y_axis_is_flipped_so_up_is_positive(self):
        pose = normalize_pose_window(make_window(2))
        # Shoulders are above the hips in the real world.
        assert pose[0, LEFT_SHOULDER, 1] > 0
        assert pose[0, LEFT_KNEE, 1] < 0

    def test_low_confidence_keypoints_become_nan(self):
        window = make_window(3)
        window[:, RIGHT_WRIST, 2] = 0.05
        pose = normalize_pose_window(window, min_confidence=0.3)
        assert np.isnan(pose[:, RIGHT_WRIST, :]).all()

    def test_accepts_two_channel_input(self):
        window = make_window(3)[:, :, :2]
        assert normalize_pose_window(window).shape == (3, NUM_POSE_KEYPOINTS, 2)

    def test_bad_shape_raises(self):
        with pytest.raises(PoseFeatureError, match="expected pose window"):
            normalize_pose_window(np.zeros((4, 12, 3)))

    def test_empty_window_raises(self):
        with pytest.raises(PoseFeatureError, match="empty"):
            normalize_pose_window(np.zeros((0, NUM_POSE_KEYPOINTS, 3)))


class TestFrameFeatures:
    """The per-frame feature time series."""

    def test_shape_and_naming_agree(self):
        series = frame_features(make_window(9))
        assert series.shape == (9, NUM_FRAME_FEATURES)
        assert len(FRAME_FEATURE_NAMES) == NUM_FRAME_FEATURES

    def test_straight_arm_reads_as_180_degrees(self):
        # Shoulder (120,140) -> elbow (120,190) -> wrist (120,240): a vertical line.
        window = make_window(3, right_elbow=(120.0, 190.0), right_wrist=(120.0, 240.0))
        series = frame_features(window)
        column = FRAME_FEATURE_NAMES.index("right_elbow_angle")
        assert series[0, column] == pytest.approx(180.0, abs=1e-6)

    def test_overhead_wrist_reads_above_shoulder(self):
        """A serve has the racket hand well above the shoulders."""
        overhead = make_window(3, right_wrist=(120.0, 20.0), right_elbow=(120.0, 80.0))
        low = make_window(3)

        column = FRAME_FEATURE_NAMES.index("highest_wrist_above_shoulder")
        assert frame_features(overhead)[0, column] > 0
        assert frame_features(low)[0, column] < 0

    def test_wrist_lateral_position_separates_sides(self):
        """Forehand and backhand differ in which side the hand swings to."""
        right_side = make_window(3, right_wrist=(220.0, 200.0))
        left_side = make_window(3, right_wrist=(-20.0, 200.0))

        column = FRAME_FEATURE_NAMES.index("right_wrist_x")
        assert frame_features(right_side)[0, column] > frame_features(left_side)[0, column]

    def test_wrist_separation_is_non_negative(self):
        series = frame_features(make_window(5))
        column = FRAME_FEATURE_NAMES.index("wrist_separation")
        assert (series[:, column] >= 0).all()

    def test_features_are_position_invariant(self):
        near = frame_features(make_window(5, origin=(0.0, 0.0)))
        far = frame_features(make_window(5, origin=(700.0, 120.0), scale=0.4))
        np.testing.assert_allclose(near, far, atol=1e-6)

    def test_occluded_limb_yields_nan_not_a_wrong_number(self):
        window = make_window(4)
        window[:, [RIGHT_WRIST, RIGHT_ELBOW], 2] = 0.0
        series = frame_features(window)
        assert np.isnan(series[:, FRAME_FEATURE_NAMES.index("right_elbow_angle")]).all()


class TestImputeAndAggregate:
    """NaN filling and the fixed-length aggregate vector."""

    def test_imputation_removes_all_nan(self):
        series = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
        filled = impute_frame_features(series)
        assert np.isfinite(filled).all()

    def test_imputation_uses_column_mean(self):
        series = np.array([[1.0, 5.0], [3.0, np.nan]])
        filled = impute_frame_features(series)
        assert filled[1, 1] == pytest.approx(5.0)

    def test_all_nan_column_becomes_zero(self):
        filled = impute_frame_features(np.array([[np.nan], [np.nan]]))
        assert (filled == 0.0).all()

    def test_aggregate_length_matches_names(self):
        vector = aggregate_features(frame_features(make_window(10)))
        assert vector.shape == (NUM_AGGREGATE_FEATURES,)
        assert len(AGGREGATE_FEATURE_NAMES) == NUM_AGGREGATE_FEATURES

    def test_aggregate_is_always_finite(self):
        window = make_window(6)
        window[:, RIGHT_WRIST, 2] = 0.0  # occluded throughout
        assert np.isfinite(aggregate_features(frame_features(window))).all()

    def test_static_pose_has_zero_variance_and_velocity(self):
        vector = aggregate_features(frame_features(make_window(8)))
        for index, name in enumerate(AGGREGATE_FEATURE_NAMES):
            if name.endswith("_std") or name.startswith(("vel",)) or "_vel_" in name:
                assert vector[index] == pytest.approx(0.0, abs=1e-9), name

    def test_moving_pose_has_non_zero_velocity(self):
        poses = [
            make_pose(right_wrist=(140.0 + 12.0 * t, 200.0 - 18.0 * t)) for t in range(8)
        ]
        vector = aggregate_features(frame_features(np.stack(poses)))
        velocity_indices = [
            i for i, name in enumerate(AGGREGATE_FEATURE_NAMES) if "vel_max_abs" in name
        ]
        assert max(vector[i] for i in velocity_indices) > 0

    def test_single_frame_window_is_handled(self):
        vector = aggregate_features(frame_features(make_window(1)))
        assert vector.shape == (NUM_AGGREGATE_FEATURES,)
        assert np.isfinite(vector).all()

    def test_extract_features_matches_two_step_path(self):
        window = make_window(7)
        np.testing.assert_allclose(
            extract_features(window), aggregate_features(frame_features(window))
        )

    def test_wrong_column_count_raises(self):
        with pytest.raises(PoseFeatureError, match="expected series"):
            aggregate_features(np.zeros((5, 3)))

    def test_empty_series_raises(self):
        with pytest.raises(PoseFeatureError, match="empty"):
            aggregate_features(np.zeros((0, NUM_FRAME_FEATURES)))


class TestResampleSeries:
    """Fixed-length resampling for the CNN backend."""

    def test_upsamples_to_requested_length(self):
        assert resample_series(np.zeros((5, 12)), 16).shape == (16, 12)

    def test_downsamples_to_requested_length(self):
        assert resample_series(np.zeros((40, 12)), 16).shape == (16, 12)

    def test_preserves_endpoints(self):
        series = np.linspace(0, 1, 10).reshape(-1, 1)
        resampled = resample_series(series, 25)
        assert resampled[0, 0] == pytest.approx(0.0)
        assert resampled[-1, 0] == pytest.approx(1.0)

    def test_linear_ramp_stays_linear(self):
        series = np.linspace(0, 100, 11).reshape(-1, 1)
        resampled = resample_series(series, 21)
        np.testing.assert_allclose(resampled[:, 0], np.linspace(0, 100, 21), atol=1e-9)

    def test_single_frame_is_repeated(self):
        resampled = resample_series(np.array([[7.0, 8.0]]), 4)
        assert resampled.shape == (4, 2)
        assert (resampled == np.array([7.0, 8.0])).all()

    def test_empty_series_raises(self):
        with pytest.raises(PoseFeatureError):
            resample_series(np.zeros((0, 12)), 16)

    def test_non_positive_length_raises(self):
        with pytest.raises(PoseFeatureError, match="length must be"):
            resample_series(np.zeros((5, 12)), 0)


class TestDetectShotMoments:
    """Shot detection from ball trajectory kinematics."""

    def test_finds_a_clean_reversal(self):
        # Ball travels right, is struck at index 15, travels back left.
        positions = [(float(i * 10), 100.0) for i in range(16)]
        positions += [(float(150 - (i + 1) * 10), 100.0) for i in range(15)]

        moments = detect_shot_moments(positions, fps=30.0, min_direction_change_deg=45.0)
        assert moments, "expected to detect the reversal"
        assert min(abs(m.index - 15) for m in moments) <= 3

    def test_straight_flight_produces_no_shots(self):
        positions = [(float(i * 10), 100.0) for i in range(60)]
        assert detect_shot_moments(positions, fps=30.0) == []

    def test_stationary_ball_produces_no_shots(self):
        assert detect_shot_moments([(50.0, 50.0)] * 40, fps=30.0) == []

    def test_reversal_registers_as_one_shot_not_many(self):
        positions = [(float(i * 10), 100.0) for i in range(16)]
        positions += [(float(150 - (i + 1) * 10), 100.0) for i in range(15)]
        moments = detect_shot_moments(
            positions, fps=30.0, min_direction_change_deg=45.0, min_shot_interval_s=0.4
        )
        assert len(moments) == 1

    def test_two_separated_rallies_give_two_shots(self):
        segment_out = [(float(i * 10), 100.0) for i in range(20)]
        segment_back = [(float(190 - i * 10), 100.0) for i in range(1, 21)]
        segment_out2 = [(float(i * 10), 100.0) for i in range(1, 21)]
        moments = detect_shot_moments(
            segment_out + segment_back + segment_out2,
            fps=30.0,
            min_direction_change_deg=45.0,
            min_shot_interval_s=0.3,
        )
        assert len(moments) == 2

    def test_shot_moment_lands_exactly_on_the_impact_frame(self):
        """The pose window centres on this index, so an off-by-two matters."""
        positions = [(float(i * 10), 100.0) for i in range(16)]
        positions += [(float(150 - (i + 1) * 10), 100.0) for i in range(15)]
        moments = detect_shot_moments(positions, fps=30.0)
        assert len(moments) == 1
        assert moments[0].index == 15

    def test_direction_change_is_reported(self):
        positions = [(float(i * 10), 100.0) for i in range(16)]
        positions += [(float(150 - (i + 1) * 10), 100.0) for i in range(15)]
        moments = detect_shot_moments(positions, fps=30.0)
        assert moments[0].direction_change_deg > 90.0

    def test_gaps_in_the_track_are_tolerated(self):
        positions: list = [(float(i * 10), 100.0) for i in range(16)]
        positions += [(float(150 - (i + 1) * 10), 100.0) for i in range(15)]
        positions[4] = positions[5] = None
        # Should not raise, and should still find the reversal at ~15.
        moments = detect_shot_moments(positions, fps=30.0)
        assert any(abs(m.index - 15) <= 3 for m in moments)

    def test_short_track_returns_nothing(self):
        assert detect_shot_moments([(1.0, 1.0), (2.0, 2.0)], fps=30.0) == []

    def test_all_none_track_returns_nothing(self):
        assert detect_shot_moments([None] * 30, fps=30.0) == []

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be > 0"):
            detect_shot_moments([(0.0, 0.0)] * 5, fps=0.0)

    def test_results_are_chronological(self):
        rng = np.random.default_rng(7)
        positions = [tuple(p) for p in rng.normal(0, 50, (120, 2))]
        moments = detect_shot_moments(positions, fps=30.0, min_shot_interval_s=0.2)
        assert [m.index for m in moments] == sorted(m.index for m in moments)


class TestAssignStriker:
    """Attributing a shot to the nearest player."""

    def test_picks_the_nearest_player(self):
        assert assign_striker((5.0, 5.0), {1: (5.5, 5.2), 2: (5.0, 20.0)}) == 1

    def test_returns_none_when_all_players_are_far(self):
        assert assign_striker((5.0, 5.0), {1: (5.0, 20.0)}, max_distance=4.0) is None

    def test_accepts_a_player_within_the_limit(self):
        assert assign_striker((5.0, 5.0), {1: (5.0, 8.0)}, max_distance=4.0) == 1

    def test_no_players_gives_none(self):
        assert assign_striker((5.0, 5.0), {}) is None

    def test_missing_ball_gives_none(self):
        assert assign_striker(None, {1: (5.0, 5.0)}) is None

    def test_players_with_unknown_positions_are_skipped(self):
        assert assign_striker((5.0, 5.0), {1: None, 2: (6.0, 6.0)}) == 2


class TestWindowBounds:
    """Pose-window extraction bounds around a shot moment."""

    def test_window_is_centred_on_the_shot(self):
        # 0.25 s at 30 fps = 7.5 frames either side, rounding to 8.
        start, end = window_bounds(50, window_seconds=0.25, fps=30.0, total_frames=200)
        assert start == 42 and end == 59
        assert start < 50 < end

    def test_clipped_at_the_start_of_the_video(self):
        start, end = window_bounds(2, window_seconds=0.25, fps=30.0, total_frames=200)
        assert start == 0
        assert end > 2

    def test_clipped_at_the_end_of_the_video(self):
        start, end = window_bounds(198, window_seconds=0.25, fps=30.0, total_frames=200)
        assert end == 200
        assert start < 198

    def test_window_length_scales_with_fps(self):
        low = window_bounds(100, 0.25, 24.0, 500)
        high = window_bounds(100, 0.25, 60.0, 500)
        assert (high[1] - high[0]) > (low[1] - low[0])

    def test_window_is_never_empty(self):
        start, end = window_bounds(0, window_seconds=0.01, fps=30.0, total_frames=10)
        assert end > start

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be > 0"):
            window_bounds(10, 0.25, 0.0, 100)
