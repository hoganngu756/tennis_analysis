"""Tests for speed and distance calculation.

Ground truth is constructed analytically: a player moving at a known metres-per-second
across a known number of frames must produce exactly the expected km/h and metres.
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis_analysis.analytics import (
    MPS_TO_KMH,
    compute_speed_series,
    cumulative_distance,
    displacement,
    speed_kmh,
    summarise_players,
)


class TestDisplacement:
    """Point-to-point distance in court metres."""

    def test_pythagorean_triple(self):
        assert displacement((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_zero_for_identical_points(self):
        assert displacement((2.5, 7.5), (2.5, 7.5)) == pytest.approx(0.0)

    def test_is_symmetric(self):
        a, b = (1.0, 2.0), (4.0, 6.0)
        assert displacement(a, b) == pytest.approx(displacement(b, a))

    def test_none_inputs_return_none(self):
        assert displacement(None, (1.0, 1.0)) is None
        assert displacement((1.0, 1.0), None) is None
        assert displacement(None, None) is None

    def test_nan_returns_none(self):
        assert displacement((np.nan, 1.0), (2.0, 2.0)) is None


class TestSpeedConversion:
    """The metres-per-second to km/h conversion."""

    def test_one_mps_is_3_6_kmh(self):
        assert speed_kmh(1.0, 1.0) == pytest.approx(3.6)

    def test_known_sprint_speed(self):
        # 10 metres in 1.0 s = 36 km/h.
        assert speed_kmh(10.0, 1.0) == pytest.approx(36.0)

    def test_scales_with_time(self):
        assert speed_kmh(10.0, 2.0) == pytest.approx(18.0)

    def test_zero_distance_is_zero_speed(self):
        assert speed_kmh(0.0, 0.5) == pytest.approx(0.0)

    def test_non_positive_elapsed_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            speed_kmh(5.0, 0.0)
        with pytest.raises(ValueError, match="must be > 0"):
            speed_kmh(5.0, -1.0)

    def test_conversion_constant(self):
        assert MPS_TO_KMH == pytest.approx(3.6)


class TestComputeSpeedSeries:
    """Per-frame speed over a smoothing window."""

    def test_constant_velocity_recovers_exact_speed(self):
        # 0.2 m per frame at 25 fps = 5 m/s = 18 km/h.
        fps = 25.0
        positions = [(0.2 * i, 0.0) for i in range(20)]
        speeds = compute_speed_series(positions, fps, window=5)

        measured = [s for s in speeds if s is not None]
        assert measured, "expected at least some speed samples"
        for value in measured:
            assert value == pytest.approx(18.0)

    def test_stationary_player_has_zero_speed(self):
        positions = [(5.0, 10.0)] * 15
        speeds = compute_speed_series(positions, 30.0, window=5)
        for value in (s for s in speeds if s is not None):
            assert value == pytest.approx(0.0)

    def test_first_frame_has_no_speed(self):
        positions = [(float(i), 0.0) for i in range(10)]
        assert compute_speed_series(positions, 30.0, window=3)[0] is None

    def test_diagonal_motion_uses_euclidean_distance(self):
        # 0.3 m in x and 0.4 m in y per frame = 0.5 m per frame.
        fps = 10.0
        positions = [(0.3 * i, 0.4 * i) for i in range(15)]
        speeds = compute_speed_series(positions, fps, window=5)
        expected = 0.5 * fps * MPS_TO_KMH  # 18 km/h
        for value in (s for s in speeds if s is not None):
            assert value == pytest.approx(expected)

    def test_gaps_are_tolerated(self):
        positions: list = [(0.2 * i, 0.0) for i in range(20)]
        positions[7] = positions[8] = None
        speeds = compute_speed_series(positions, 25.0, window=5)

        assert speeds[7] is None and speeds[8] is None
        # Frames after the gap still resolve, using the actual elapsed time.
        assert speeds[12] == pytest.approx(18.0)

    def test_implausible_speed_is_clipped(self):
        positions = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (500.0, 0.0)]
        speeds = compute_speed_series(positions, 30.0, window=1, max_speed_kmh=40.0)
        assert speeds[3] is None

    def test_speed_just_below_limit_is_kept(self):
        # 0.3 m per frame at 30 fps = 9 m/s = 32.4 km/h, under a 40 km/h cap.
        positions = [(0.3 * i, 0.0) for i in range(10)]
        speeds = compute_speed_series(positions, 30.0, window=3, max_speed_kmh=40.0)
        assert speeds[-1] == pytest.approx(32.4)

    def test_all_none_positions_give_all_none(self):
        assert compute_speed_series([None] * 10, 30.0) == [None] * 10

    def test_empty_input(self):
        assert compute_speed_series([], 30.0) == []

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be > 0"):
            compute_speed_series([(0.0, 0.0)], 0.0)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            compute_speed_series([(0.0, 0.0)], 30.0, window=0)


class TestCumulativeDistance:
    """Running total of ground covered."""

    def test_straight_line_total(self):
        # 10 steps of 0.5 m = 5 m total.
        positions = [(0.5 * i, 0.0) for i in range(11)]
        totals = cumulative_distance(positions, fps=30.0)
        assert totals[-1] == pytest.approx(5.0)

    def test_is_monotonic_non_decreasing(self):
        rng = np.random.default_rng(0)
        positions = [(float(x), float(y)) for x, y in rng.normal(0, 0.1, (50, 2)).cumsum(axis=0)]
        totals = cumulative_distance(positions, fps=30.0)
        assert all(b >= a for a, b in zip(totals, totals[1:]))

    def test_stationary_player_covers_nothing(self):
        totals = cumulative_distance([(3.0, 4.0)] * 20, fps=30.0)
        assert totals[-1] == pytest.approx(0.0)

    def test_returning_to_start_still_accumulates(self):
        """Distance covered is path length, not displacement."""
        positions = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
        totals = cumulative_distance(positions, fps=30.0)
        assert totals[-1] == pytest.approx(4.0)

    def test_length_matches_input(self):
        positions = [(float(i), 0.0) for i in range(17)]
        assert len(cumulative_distance(positions, fps=30.0)) == 17

    def test_gaps_hold_the_previous_total(self):
        positions: list = [(0.0, 0.0), (1.0, 0.0), None, None, (2.0, 0.0)]
        totals = cumulative_distance(positions, fps=1.0, max_speed_kmh=None)
        assert totals[1] == pytest.approx(1.0)
        assert totals[2] == pytest.approx(1.0)
        assert totals[3] == pytest.approx(1.0)
        assert totals[4] == pytest.approx(2.0)

    def test_teleport_from_id_switch_is_excluded(self):
        # A 30 m jump in one frame at 30 fps implies ~3240 km/h — clearly an ID switch.
        positions = [(0.0, 0.0), (0.1, 0.0), (30.0, 0.0), (30.1, 0.0)]
        totals = cumulative_distance(positions, fps=30.0, max_speed_kmh=40.0)
        assert totals[-1] == pytest.approx(0.2)

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be > 0"):
            cumulative_distance([(0.0, 0.0)], fps=0.0)


class TestPixelToSpeedIntegration:
    """The full seam: bounding box -> feet -> homography -> km/h.

    Each piece is unit-tested elsewhere; this pins the composition, which is where a
    wrong anchor point or a transposed projection would actually show up.
    """

    @staticmethod
    def _camera():
        """A metre -> pixel homography for a raised camera behind the baseline."""
        import cv2

        from tennis_analysis.court_keypoints.geometry import COURT_LENGTH, DOUBLES_WIDTH

        court = np.array(
            [[0, 0], [DOUBLES_WIDTH, 0], [DOUBLES_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]],
            dtype=np.float64,
        )
        image = np.array([[760, 300], [1160, 300], [1600, 950], [320, 950]], dtype=np.float64)
        matrix, _ = cv2.findHomography(court, image)
        return matrix

    def test_player_running_at_known_speed_measures_correctly(self):
        """A player crossing 4 m/s must read as 14.4 km/h through the whole chain."""
        from tennis_analysis.court_keypoints.geometry import (
            COURT_KEYPOINTS_M,
            CourtHomography,
        )
        from tennis_analysis.types import BBox

        camera = self._camera()

        def to_pixels(points):
            homogeneous = np.hstack([points, np.ones((len(points), 1))])
            projected = homogeneous @ camera.T
            return projected[:, :2] / projected[:, 2:3]

        homography = CourtHomography.from_keypoints(to_pixels(COURT_KEYPOINTS_M))

        fps = 25.0
        speed_mps = 4.0
        step = speed_mps / fps  # metres per frame

        positions = []
        for i in range(20):
            # Ground-truth court position, walking across the baseline area.
            ground_truth = np.array([[2.0 + step * i, 20.0]])
            feet_px = to_pixels(ground_truth)[0]

            # Build a plausible box whose *bottom centre* is that ground point.
            box = BBox(feet_px[0] - 30, feet_px[1] - 160, feet_px[0] + 30, feet_px[1])
            court = homography.to_court(np.array(box.feet))
            positions.append((float(court[0]), float(court[1])))

        speeds = compute_speed_series(positions, fps, window=5, max_speed_kmh=40.0)
        measured = [s for s in speeds if s is not None]

        assert measured
        expected_kmh = speed_mps * MPS_TO_KMH  # 14.4
        for value in measured:
            assert value == pytest.approx(expected_kmh, abs=0.05)

    def test_distance_covered_matches_ground_truth(self):
        """Walking a known 3 m path must accumulate 3 m of distance."""
        from tennis_analysis.court_keypoints.geometry import (
            COURT_KEYPOINTS_M,
            CourtHomography,
        )
        from tennis_analysis.types import BBox

        camera = self._camera()

        def to_pixels(points):
            homogeneous = np.hstack([points, np.ones((len(points), 1))])
            projected = homogeneous @ camera.T
            return projected[:, :2] / projected[:, 2:3]

        homography = CourtHomography.from_keypoints(to_pixels(COURT_KEYPOINTS_M))

        positions = []
        for i in range(31):
            feet_px = to_pixels(np.array([[3.0, 15.0 + 0.1 * i]]))[0]
            box = BBox(feet_px[0] - 25, feet_px[1] - 150, feet_px[0] + 25, feet_px[1])
            court = homography.to_court(np.array(box.feet))
            positions.append((float(court[0]), float(court[1])))

        totals = cumulative_distance(positions, fps=30.0, max_speed_kmh=40.0)
        assert totals[-1] == pytest.approx(3.0, abs=0.01)

    def test_box_centre_would_be_wrong(self):
        """Anchoring on the box centre instead of the feet lands deep in the court."""
        from tennis_analysis.court_keypoints.geometry import (
            COURT_KEYPOINTS_M,
            CourtHomography,
        )
        from tennis_analysis.types import BBox

        camera = self._camera()

        def to_pixels(points):
            homogeneous = np.hstack([points, np.ones((len(points), 1))])
            projected = homogeneous @ camera.T
            return projected[:, :2] / projected[:, 2:3]

        homography = CourtHomography.from_keypoints(to_pixels(COURT_KEYPOINTS_M))

        truth = np.array([[5.0, 20.0]])
        feet_px = to_pixels(truth)[0]
        box = BBox(feet_px[0] - 30, feet_px[1] - 170, feet_px[0] + 30, feet_px[1])

        from_feet = homography.to_court(np.array(box.feet))
        from_centre = homography.to_court(np.array(box.center))

        assert from_feet[1] == pytest.approx(20.0, abs=1e-6)
        # The centre projects metres away — this is why `feet` is the anchor.
        assert abs(from_centre[1] - 20.0) > 1.0


class TestSummarisePlayers:
    """The per-player rollup table."""

    def test_reports_totals_and_extremes(self):
        distances = {1: [0.0, 1.0, 2.5], 2: [0.0, 0.5, 0.9]}
        speeds = {1: [None, 10.0, 20.0], 2: [None, 5.0, 7.0]}
        summary = summarise_players(distances, speeds).set_index("player")

        assert summary.loc[1, "total_distance_m"] == pytest.approx(2.5)
        assert summary.loc[1, "max_speed_kmh"] == pytest.approx(20.0)
        assert summary.loc[1, "avg_speed_kmh"] == pytest.approx(15.0)
        assert summary.loc[2, "total_distance_m"] == pytest.approx(0.9)

    def test_handles_player_with_no_speed_samples(self):
        summary = summarise_players({1: [0.0, 0.0]}, {1: [None, None]}).set_index("player")
        assert summary.loc[1, "avg_speed_kmh"] == pytest.approx(0.0)
        assert summary.loc[1, "max_speed_kmh"] == pytest.approx(0.0)

    def test_empty_input_gives_empty_frame(self):
        assert summarise_players({}, {}).empty
