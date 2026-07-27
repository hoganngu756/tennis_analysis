"""Tests for the ball Kalman filter: smoothing, interpolation and gating."""

from __future__ import annotations

import numpy as np
import pytest

from tennis_analysis.config import KalmanConfig
from tennis_analysis.tracking import BallKalmanTracker, smooth_ball_track


@pytest.fixture
def config() -> KalmanConfig:
    """A filter tuned to trust detections fairly strongly."""
    return KalmanConfig(
        process_noise=1.0, measurement_noise=1.0, max_age=5, max_gate_distance=150.0
    )


class TestInitialisation:
    """Track startup behaviour."""

    def test_no_detections_yields_nothing(self, config):
        tracker = BallKalmanTracker(config)
        assert tracker.update(None) is None
        assert not tracker.is_alive

    def test_first_detection_is_passed_through_exactly(self, config):
        tracker = BallKalmanTracker(config)
        result = tracker.update((100.0, 200.0), 0.9)
        assert result is not None
        assert result.position == pytest.approx((100.0, 200.0))
        assert not result.interpolated

    def test_track_is_alive_after_first_detection(self, config):
        tracker = BallKalmanTracker(config)
        tracker.update((100.0, 200.0))
        assert tracker.is_alive


class TestSmoothing:
    """Noise suppression on a known trajectory."""

    def test_tracks_constant_velocity(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(25):
            result = tracker.update((10.0 * i, 5.0 * i))
        assert result.position[0] == pytest.approx(240.0, abs=2.0)
        assert result.position[1] == pytest.approx(120.0, abs=2.0)

    def test_velocity_estimate_converges(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(30):
            tracker.update((10.0 * i, 0.0))
        vx, vy = tracker.velocity
        assert vx == pytest.approx(10.0, abs=1.0)
        assert vy == pytest.approx(0.0, abs=1.0)

    def test_smoothing_reduces_measurement_noise(self, config):
        """Filtered output must sit closer to truth than the raw detections do."""
        rng = np.random.default_rng(0)
        truth = [(10.0 * i, 100.0) for i in range(60)]
        noisy = [(x + rng.normal(0, 6), y + rng.normal(0, 6)) for x, y in truth]

        tracker = BallKalmanTracker(config)
        filtered = [tracker.update(point).position for point in noisy]

        # Skip the burn-in while the filter is still converging.
        raw_error = np.mean([np.hypot(n[0] - t[0], n[1] - t[1])
                             for n, t in zip(noisy[15:], truth[15:])])
        filtered_error = np.mean([np.hypot(f[0] - t[0], f[1] - t[1])
                                  for f, t in zip(filtered[15:], truth[15:])])
        assert filtered_error < raw_error


class TestInterpolation:
    """Prediction through occlusions."""

    def test_gap_is_flagged_as_interpolated(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(10):
            tracker.update((10.0 * i, 0.0))
        result = tracker.update(None)
        assert result is not None
        assert result.interpolated

    def test_prediction_continues_the_trajectory(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(20):
            tracker.update((10.0 * i, 0.0))
        # Last observation was at x=190; one frame on should be near x=200.
        result = tracker.update(None)
        assert result.position[0] == pytest.approx(200.0, abs=8.0)

    def test_interpolated_frames_carry_zero_confidence(self, config):
        tracker = BallKalmanTracker(config)
        tracker.update((0.0, 0.0), 0.9)
        tracker.update((10.0, 0.0), 0.9)
        assert tracker.update(None).confidence == 0.0

    def test_track_dies_after_max_age(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(10):
            tracker.update((10.0 * i, 0.0))
        for _ in range(config.max_age):
            assert tracker.update(None) is not None
        assert tracker.update(None) is None
        assert not tracker.is_alive

    def test_track_reseeds_cleanly_after_death(self, config):
        tracker = BallKalmanTracker(config)
        tracker.update((0.0, 0.0))
        for _ in range(config.max_age + 1):
            tracker.update(None)

        # A detection far from the stale prediction must still restart the track.
        result = tracker.update((900.0, 700.0))
        assert result is not None
        assert result.position == pytest.approx((900.0, 700.0))

    def test_short_gap_then_recovery(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(15):
            tracker.update((10.0 * i, 0.0))
        tracker.update(None)
        tracker.update(None)
        result = tracker.update((170.0, 0.0))
        assert not result.interpolated
        assert result.position[0] == pytest.approx(170.0, abs=12.0)


class TestGating:
    """Rejection of implausible detections."""

    def test_outlier_far_from_prediction_is_rejected(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(15):
            tracker.update((10.0 * i, 0.0))
        # A detection 900 px away is a false positive, not the ball.
        result = tracker.update((1200.0, 900.0))
        assert result.interpolated
        assert result.position[0] < 300.0

    def test_detection_inside_the_gate_is_accepted(self, config):
        tracker = BallKalmanTracker(config)
        for i in range(15):
            tracker.update((10.0 * i, 0.0))
        result = tracker.update((165.0, 20.0))
        assert not result.interpolated


class TestSmoothBallTrack:
    """The sequence-level convenience wrapper."""

    def test_output_length_matches_input(self, config):
        measurements = [(float(i), 0.0) for i in range(30)]
        assert len(smooth_ball_track(measurements, config)) == 30

    def test_leading_gap_yields_none(self, config):
        measurements = [None, None, (10.0, 10.0), (12.0, 10.0)]
        results = smooth_ball_track(measurements, config)
        assert results[0] is None and results[1] is None
        assert results[2] is not None

    def test_interior_gaps_are_filled(self, config):
        measurements: list = [(10.0 * i, 0.0) for i in range(20)]
        measurements[10] = measurements[11] = None
        results = smooth_ball_track(measurements, config)
        assert results[10] is not None and results[10].interpolated
        assert results[11] is not None and results[11].interpolated

    def test_all_none_gives_all_none(self, config):
        assert smooth_ball_track([None] * 10, config) == [None] * 10

    def test_empty_input(self, config):
        assert smooth_ball_track([], config) == []

    def test_confidences_are_carried_through(self, config):
        measurements = [(float(i), 0.0) for i in range(5)]
        results = smooth_ball_track(measurements, config, confidences=[0.7] * 5)
        assert results[0].confidence == pytest.approx(0.7)

    def test_mismatched_confidence_length_raises(self, config):
        with pytest.raises(ValueError, match="length"):
            smooth_ball_track([(1.0, 1.0)] * 5, config, confidences=[0.5] * 3)
