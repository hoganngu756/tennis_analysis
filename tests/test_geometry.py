"""Tests for the court homography transform.

Strategy: build a *known* synthetic camera by projecting the 14 reference court points
through a hand-constructed homography. Fitting on those projections must recover the
original court coordinates, so ground truth is exact rather than eyeballed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tennis_analysis.court_keypoints.geometry import (
    COURT_KEYPOINTS_M,
    COURT_LENGTH,
    DOUBLES_WIDTH,
    NUM_KEYPOINTS,
    SERVICE_LINE_FAR_Y,
    SERVICE_LINE_NEAR_Y,
    SINGLES_INSET,
    SINGLES_WIDTH,
    CourtHomography,
    HomographyError,
    is_inside_court,
)


@pytest.fixture
def camera() -> np.ndarray:
    """A metre -> pixel homography approximating a raised camera behind the baseline."""
    # Court corners mapped to a plausible trapezoid in a 1920x1080 frame: the far
    # baseline is narrower and higher up, exactly as a real broadcast view renders it.
    court_corners = np.array(
        [[0, 0], [DOUBLES_WIDTH, 0], [DOUBLES_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]],
        dtype=np.float64,
    )
    image_corners = np.array(
        [[760, 300], [1160, 300], [1600, 950], [320, 950]], dtype=np.float64
    )
    matrix, _ = cv2.findHomography(court_corners, image_corners)
    return matrix


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to ``(N, 2)`` points."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


@pytest.fixture
def pixel_keypoints(camera: np.ndarray) -> np.ndarray:
    """The 14 reference keypoints as they would appear in the synthetic camera."""
    return project(camera, COURT_KEYPOINTS_M)


class TestCourtConstants:
    """The reference layout must match real ITF court dimensions."""

    def test_keypoint_count(self):
        assert COURT_KEYPOINTS_M.shape == (NUM_KEYPOINTS, 2)
        assert NUM_KEYPOINTS == 14

    def test_itf_dimensions(self):
        assert COURT_LENGTH == pytest.approx(23.77)
        assert DOUBLES_WIDTH == pytest.approx(10.97)
        assert SINGLES_WIDTH == pytest.approx(8.23)

    def test_singles_inset_is_symmetric(self):
        assert SINGLES_INSET == pytest.approx(1.37)
        assert DOUBLES_WIDTH - 2 * SINGLES_INSET == pytest.approx(SINGLES_WIDTH)

    def test_service_lines_straddle_the_net(self):
        net = COURT_LENGTH / 2
        assert SERVICE_LINE_NEAR_Y == pytest.approx(net - 6.40)
        assert SERVICE_LINE_FAR_Y == pytest.approx(net + 6.40)
        assert SERVICE_LINE_NEAR_Y == pytest.approx(5.485)
        assert SERVICE_LINE_FAR_Y == pytest.approx(18.285)

    def test_corners_span_the_full_court(self):
        assert COURT_KEYPOINTS_M[:, 0].min() == pytest.approx(0.0)
        assert COURT_KEYPOINTS_M[:, 0].max() == pytest.approx(DOUBLES_WIDTH)
        assert COURT_KEYPOINTS_M[:, 1].min() == pytest.approx(0.0)
        assert COURT_KEYPOINTS_M[:, 1].max() == pytest.approx(COURT_LENGTH)


class TestHomographyFit:
    """Fitting on exact projections must recover court coordinates exactly."""

    def test_recovers_reference_keypoints(self, pixel_keypoints):
        homography = CourtHomography.from_keypoints(pixel_keypoints)
        recovered = homography.to_court(pixel_keypoints)
        np.testing.assert_allclose(recovered, COURT_KEYPOINTS_M, atol=1e-6)

    def test_reprojection_error_is_negligible_on_clean_input(self, pixel_keypoints):
        homography = CourtHomography.from_keypoints(pixel_keypoints)
        assert homography.reprojection_error < 1e-6

    def test_round_trip_pixels_to_court_and_back(self, pixel_keypoints):
        homography = CourtHomography.from_keypoints(pixel_keypoints)
        sample = np.array([[900.0, 600.0], [1200.0, 800.0], [500.0, 940.0]])
        np.testing.assert_allclose(
            homography.to_pixels(homography.to_court(sample)), sample, atol=1e-6
        )

    def test_single_point_input_returns_single_point(self, pixel_keypoints):
        homography = CourtHomography.from_keypoints(pixel_keypoints)
        result = homography.to_court(np.array([900.0, 600.0]))
        assert result.shape == (2,)

    def test_known_court_landmark_maps_correctly(self, camera, pixel_keypoints):
        """The net centre should land at (half width, half length)."""
        homography = CourtHomography.from_keypoints(pixel_keypoints)
        net_center_px = project(camera, np.array([[DOUBLES_WIDTH / 2, COURT_LENGTH / 2]]))
        recovered = homography.to_court(net_center_px)[0]
        assert recovered[0] == pytest.approx(DOUBLES_WIDTH / 2, abs=1e-6)
        assert recovered[1] == pytest.approx(COURT_LENGTH / 2, abs=1e-6)

    def test_distances_are_preserved_in_metres(self, camera, pixel_keypoints):
        """A 5 m court-space segment must measure 5 m after inverse projection."""
        homography = CourtHomography.from_keypoints(pixel_keypoints)
        start_m = np.array([[2.0, 8.0]])
        end_m = np.array([[2.0, 13.0]])
        start_px = project(camera, start_m)
        end_px = project(camera, end_m)

        recovered = homography.to_court(np.vstack([start_px, end_px]))
        measured = float(np.linalg.norm(recovered[1] - recovered[0]))
        assert measured == pytest.approx(5.0, abs=1e-4)


class TestHomographyRobustness:
    """Confidence gating and error handling."""

    def test_low_confidence_keypoints_are_excluded(self, pixel_keypoints):
        corrupted = pixel_keypoints.copy()
        corrupted[3] = [0.0, 0.0]  # nonsense position
        confidences = np.ones(NUM_KEYPOINTS)
        confidences[3] = 0.1  # ... but flagged as unreliable

        homography = CourtHomography.from_keypoints(
            corrupted, confidences, min_confidence=0.5
        )
        # The bad point was dropped, so the remaining 13 still fit perfectly.
        assert homography.reprojection_error < 1e-6

    def test_fit_succeeds_with_exactly_four_confident_points(self, pixel_keypoints):
        confidences = np.zeros(NUM_KEYPOINTS)
        confidences[[0, 1, 2, 3]] = 1.0
        homography = CourtHomography.from_keypoints(pixel_keypoints, confidences)
        assert homography.reprojection_error < 1e-6

    def test_fewer_than_four_points_raises(self, pixel_keypoints):
        confidences = np.zeros(NUM_KEYPOINTS)
        confidences[[0, 1, 2]] = 1.0
        with pytest.raises(HomographyError, match="need >=4"):
            CourtHomography.from_keypoints(pixel_keypoints, confidences)

    def test_nan_keypoints_are_dropped(self, pixel_keypoints):
        corrupted = pixel_keypoints.copy()
        corrupted[5] = [np.nan, np.nan]
        homography = CourtHomography.from_keypoints(corrupted)
        assert homography.reprojection_error < 1e-6

    def test_wrong_shape_raises(self):
        with pytest.raises(HomographyError, match="expected"):
            CourtHomography.from_keypoints(np.zeros((10, 2)))

    def test_collinear_points_raise(self):
        """Points on a single line cannot define a plane-to-plane mapping."""
        collinear = np.stack(
            [np.linspace(0, 100, NUM_KEYPOINTS), np.linspace(0, 100, NUM_KEYPOINTS)],
            axis=1,
        )
        with pytest.raises(HomographyError):
            CourtHomography.from_keypoints(collinear)

    def test_noisy_keypoints_give_bounded_error(self, pixel_keypoints):
        rng = np.random.default_rng(42)
        noisy = pixel_keypoints + rng.normal(0, 2.0, pixel_keypoints.shape)
        homography = CourtHomography.from_keypoints(noisy)
        # A couple of pixels of keypoint noise should stay well under a metre on court.
        assert homography.reprojection_error < 1.0


class TestIsInsideCourt:
    """Court-boundary filtering used to reject ball kids and officials."""

    def test_court_centre_is_inside(self):
        assert is_inside_court(np.array([[DOUBLES_WIDTH / 2, COURT_LENGTH / 2]]))[0]

    def test_corners_are_inside(self):
        assert is_inside_court(COURT_KEYPOINTS_M, margin=0.0).all()

    def test_far_spectator_is_outside(self):
        assert not is_inside_court(np.array([[-20.0, 12.0]]), margin=2.0)[0]

    def test_margin_admits_players_behind_the_baseline(self):
        just_behind = np.array([[5.0, COURT_LENGTH + 1.5]])
        assert not is_inside_court(just_behind, margin=0.0)[0]
        assert is_inside_court(just_behind, margin=2.0)[0]

    def test_nan_is_outside(self):
        assert not is_inside_court(np.array([[np.nan, 5.0]]))[0]

    def test_vectorised_over_many_points(self):
        points = np.array([[5.0, 12.0], [-30.0, 12.0], [5.0, 60.0]])
        np.testing.assert_array_equal(
            is_inside_court(points, margin=2.0), [True, False, False]
        )
