"""Stage 3 (part 2) — Kalman smoothing and gap interpolation for the ball track.

The ball is small, fast and frequently occluded by players or lost against the crowd.
A constant-acceleration Kalman filter smooths the noisy per-frame detections and
predicts through the gaps, so downstream speed and shot-moment logic sees a continuous
trajectory instead of a sequence of holes.
"""

from __future__ import annotations

import numpy as np

from .config import KalmanConfig
from .types import BallDetection

#: State vector layout: [x, y, vx, vy, ax, ay].
_STATE_DIM = 6
_MEASUREMENT_DIM = 2


class BallKalmanTracker:
    """Constant-acceleration Kalman filter over ball detections.

    Call :meth:`update` once per processed frame with the raw detection (or ``None``
    when the detector found nothing). The tracker returns a smoothed position, flagged
    as interpolated when it came from prediction alone.
    """

    def __init__(self, config: KalmanConfig, dt: float = 1.0) -> None:
        """Initialise the filter.

        Args:
            config: Noise and gating parameters.
            dt: Time step between processed frames, in frame units. Left at 1.0 the
                filter works in pixels-per-frame; speeds are converted to real units
                downstream via the homography and FPS.
        """
        self.config = config
        self.dt = dt

        # Constant-acceleration transition.
        self._F = np.eye(_STATE_DIM)
        self._F[0, 2] = self._F[1, 3] = dt
        self._F[2, 4] = self._F[3, 5] = dt
        self._F[0, 4] = self._F[1, 5] = 0.5 * dt**2

        # We only ever observe position.
        self._H = np.zeros((_MEASUREMENT_DIM, _STATE_DIM))
        self._H[0, 0] = self._H[1, 1] = 1.0

        self._Q = np.eye(_STATE_DIM) * config.process_noise
        self._R = np.eye(_MEASUREMENT_DIM) * config.measurement_noise

        self._state: np.ndarray | None = None
        self._covariance = np.eye(_STATE_DIM) * 1000.0
        self._frames_since_detection = 0

    @property
    def is_alive(self) -> bool:
        """Whether the tracker currently holds a usable estimate."""
        return (
            self._state is not None
            and self._frames_since_detection <= self.config.max_age
        )

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Current velocity estimate in pixels per frame, or ``None`` if not tracking."""
        if self._state is None:
            return None
        return (float(self._state[2]), float(self._state[3]))

    def update(
        self,
        measurement: tuple[float, float] | None,
        confidence: float = 0.0,
    ) -> BallDetection | None:
        """Advance the filter one frame.

        Args:
            measurement: Raw detected ball centre in pixels, or ``None`` if the
                detector produced nothing this frame.
            confidence: Detector confidence for ``measurement``.

        Returns:
            The smoothed :class:`BallDetection`, or ``None`` if the track is dead
            (no detection for longer than ``max_age``, or never initialised).
        """
        if self._state is None:
            if measurement is None:
                return None
            self._initialise(measurement)
            return BallDetection(
                position=measurement, confidence=confidence, interpolated=False
            )

        # Predict.
        self._state = self._F @ self._state
        self._covariance = self._F @ self._covariance @ self._F.T + self._Q

        accepted = measurement is not None and self._within_gate(measurement)
        if accepted:
            self._correct(np.asarray(measurement, dtype=np.float64))
            self._frames_since_detection = 0
        else:
            self._frames_since_detection += 1

        if self._frames_since_detection > self.config.max_age:
            # Track is stale. Reset so a fresh detection re-seeds cleanly rather than
            # being gated out against a long-obsolete prediction.
            self._state = None
            self._covariance = np.eye(_STATE_DIM) * 1000.0
            return None

        return BallDetection(
            position=(float(self._state[0]), float(self._state[1])),
            confidence=confidence if accepted else 0.0,
            interpolated=not accepted,
        )

    def _initialise(self, measurement: tuple[float, float]) -> None:
        """Seed the state from the first detection, with zero velocity/acceleration."""
        self._state = np.zeros(_STATE_DIM)
        self._state[0], self._state[1] = measurement
        self._covariance = np.eye(_STATE_DIM) * 1000.0
        self._frames_since_detection = 0

    def _within_gate(self, measurement: tuple[float, float]) -> bool:
        """Reject detections implausibly far from the predicted position."""
        assert self._state is not None
        predicted = self._state[:2]
        distance = float(np.linalg.norm(np.asarray(measurement) - predicted))
        return distance <= self.config.max_gate_distance

    def _correct(self, measurement: np.ndarray) -> None:
        """Standard Kalman measurement update."""
        assert self._state is not None
        innovation = measurement - self._H @ self._state
        S = self._H @ self._covariance @ self._H.T + self._R
        gain = self._covariance @ self._H.T @ np.linalg.inv(S)
        self._state = self._state + gain @ innovation
        identity = np.eye(_STATE_DIM)
        self._covariance = (identity - gain @ self._H) @ self._covariance


def smooth_ball_track(
    measurements: list[tuple[float, float] | None],
    config: KalmanConfig,
    confidences: list[float] | None = None,
) -> list[BallDetection | None]:
    """Run the Kalman tracker over a full sequence of per-frame ball detections.

    Args:
        measurements: One entry per frame — the detected ball centre, or ``None``.
        config: Kalman tuning parameters.
        confidences: Optional per-frame detector confidences, same length.

    Returns:
        One :class:`BallDetection` per frame (``None`` where the track was dead).

    Raises:
        ValueError: If ``confidences`` is given with a different length.
    """
    if confidences is not None and len(confidences) != len(measurements):
        raise ValueError(
            f"confidences length {len(confidences)} != measurements length "
            f"{len(measurements)}"
        )

    tracker = BallKalmanTracker(config)
    results: list[BallDetection | None] = []
    for i, measurement in enumerate(measurements):
        confidence = confidences[i] if confidences is not None else 0.0
        results.append(tracker.update(measurement, confidence))
    return results
