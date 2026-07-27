"""Court geometry: the 14-keypoint reference layout and pixel->meter homography.

All real-world coordinates are in metres, in a court frame whose origin sits at the
top-left doubles corner, with ``x`` running across the court width and ``y`` running
along the court length toward the bottom baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- ITF court dimensions (metres) -------------------------------------------------
COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
#: Distance from the net to either service line.
SERVICE_LINE_FROM_NET = 6.40

#: Inset of each singles sideline from the corresponding doubles sideline.
SINGLES_INSET = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2  # 1.37
NET_Y = COURT_LENGTH / 2  # 11.885
CENTER_X = DOUBLES_WIDTH / 2  # 5.485
SERVICE_LINE_NEAR_Y = NET_Y - SERVICE_LINE_FROM_NET  # 5.485
SERVICE_LINE_FAR_Y = NET_Y + SERVICE_LINE_FROM_NET  # 18.285

#: Human-readable name of each of the 14 keypoints, in canonical index order.
KEYPOINT_NAMES: tuple[str, ...] = (
    "doubles_top_left",
    "doubles_top_right",
    "doubles_bottom_left",
    "doubles_bottom_right",
    "singles_top_left",
    "singles_top_right",
    "singles_bottom_left",
    "singles_bottom_right",
    "service_near_left",
    "service_near_right",
    "service_far_left",
    "service_far_right",
    "center_t_near",
    "center_t_far",
)

#: Canonical court-space location (metres) of each keypoint in ``KEYPOINT_NAMES``.
COURT_KEYPOINTS_M: np.ndarray = np.array(
    [
        [0.0, 0.0],
        [DOUBLES_WIDTH, 0.0],
        [0.0, COURT_LENGTH],
        [DOUBLES_WIDTH, COURT_LENGTH],
        [SINGLES_INSET, 0.0],
        [DOUBLES_WIDTH - SINGLES_INSET, 0.0],
        [SINGLES_INSET, COURT_LENGTH],
        [DOUBLES_WIDTH - SINGLES_INSET, COURT_LENGTH],
        [SINGLES_INSET, SERVICE_LINE_NEAR_Y],
        [DOUBLES_WIDTH - SINGLES_INSET, SERVICE_LINE_NEAR_Y],
        [SINGLES_INSET, SERVICE_LINE_FAR_Y],
        [DOUBLES_WIDTH - SINGLES_INSET, SERVICE_LINE_FAR_Y],
        [CENTER_X, SERVICE_LINE_NEAR_Y],
        [CENTER_X, SERVICE_LINE_FAR_Y],
    ],
    dtype=np.float64,
)

NUM_KEYPOINTS = len(KEYPOINT_NAMES)


class HomographyError(RuntimeError):
    """Raised when a homography cannot be estimated from the supplied keypoints."""


@dataclass(frozen=True)
class CourtHomography:
    """Maps image pixels to court metres (and back) for a single camera view.

    Attributes:
        matrix: 3x3 pixel -> metre homography.
        inverse: 3x3 metre -> pixel homography.
        reprojection_error: Mean euclidean reprojection error in pixels over the
            keypoints used to fit the homography. Useful as a per-frame quality gate.
    """

    matrix: np.ndarray
    inverse: np.ndarray
    reprojection_error: float

    @classmethod
    def from_keypoints(
        cls,
        pixel_keypoints: np.ndarray,
        confidences: np.ndarray | None = None,
        min_confidence: float = 0.5,
        ransac_threshold: float = 5.0,
    ) -> "CourtHomography":
        """Fit a homography from detected court keypoints.

        Args:
            pixel_keypoints: ``(14, 2)`` array of detected keypoint pixel coordinates,
                ordered to match :data:`KEYPOINT_NAMES`.
            confidences: Optional ``(14,)`` array of per-keypoint confidences. Points
                scoring below ``min_confidence`` are excluded from the fit.
            min_confidence: Confidence floor for including a keypoint.
            ransac_threshold: RANSAC inlier threshold in pixels.

        Returns:
            The fitted :class:`CourtHomography`.

        Raises:
            HomographyError: If fewer than 4 usable keypoints remain, or if OpenCV
                fails to find a valid (invertible) homography.
        """
        pixel_keypoints = np.asarray(pixel_keypoints, dtype=np.float64)
        if pixel_keypoints.shape != (NUM_KEYPOINTS, 2):
            raise HomographyError(
                f"expected ({NUM_KEYPOINTS}, 2) keypoints, got {pixel_keypoints.shape}"
            )

        mask = np.isfinite(pixel_keypoints).all(axis=1)
        if confidences is not None:
            mask &= np.asarray(confidences, dtype=np.float64) >= min_confidence

        src = pixel_keypoints[mask]
        dst = COURT_KEYPOINTS_M[mask]
        if len(src) < 4:
            raise HomographyError(
                f"need >=4 confident keypoints to fit a homography, got {len(src)}"
            )

        matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
        if matrix is None:
            raise HomographyError("cv2.findHomography failed to converge")

        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate fit
            raise HomographyError("estimated homography is singular") from exc

        projected = _apply_homography(matrix, src)
        error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
        return cls(matrix=matrix, inverse=inverse, reprojection_error=error)

    def to_court(self, points: np.ndarray) -> np.ndarray:
        """Project image points (pixels) into court space (metres).

        Args:
            points: ``(N, 2)`` array of pixel coordinates, or a single ``(2,)`` point.

        Returns:
            Array of the same shape holding court-space metres.
        """
        return _apply_homography(self.matrix, points)

    def to_pixels(self, points: np.ndarray) -> np.ndarray:
        """Project court-space points (metres) back into image space (pixels).

        Args:
            points: ``(N, 2)`` array of court coordinates, or a single ``(2,)`` point.

        Returns:
            Array of the same shape holding pixel coordinates.
        """
        return _apply_homography(self.inverse, points)


def _apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to ``(N, 2)`` or ``(2,)`` points, handling reshaping."""
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    if pts.shape[1] != 2:
        raise ValueError(f"expected points with shape (N, 2), got {pts.shape}")

    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    # Points on the horizon project to w ~= 0; emit NaN rather than a huge bogus value.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(w) < 1e-9, np.nan, projected[:, :2] / w)
    return result[0] if single else result


def is_inside_court(court_points: np.ndarray, margin: float = 2.0) -> np.ndarray:
    """Test whether court-space points fall within the court plus a margin.

    Used to reject ball kids, umpires and spectators, who sit well outside the
    playing surface once projected into court space.

    Args:
        court_points: ``(N, 2)`` array of court coordinates in metres.
        margin: Slack in metres added around the doubles court on all sides.

    Returns:
        ``(N,)`` boolean array. NaN coordinates yield ``False``.
    """
    pts = np.atleast_2d(np.asarray(court_points, dtype=np.float64))
    inside = (
        (pts[:, 0] >= -margin)
        & (pts[:, 0] <= DOUBLES_WIDTH + margin)
        & (pts[:, 1] >= -margin)
        & (pts[:, 1] <= COURT_LENGTH + margin)
    )
    return inside & np.isfinite(pts).all(axis=1)
