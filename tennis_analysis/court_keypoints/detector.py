"""Stage 4 — court keypoint inference and homography maintenance."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..config import CourtConfig
from .geometry import NUM_KEYPOINTS, CourtHomography, HomographyError
from .model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CourtKeypointNet,
    build_resnet_fc_regressor,
)

logger = logging.getLogger(__name__)


def select_device() -> torch.device:
    """Pick the best available torch device (CUDA, then Apple MPS, then CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def identify_architecture(state_dict: dict[str, torch.Tensor]) -> tuple[str, int]:
    """Fingerprint a checkpoint's architecture from its parameter names and shapes.

    Lets the pipeline consume both this project's dual-head model and the single-head
    ResNet regressors that circulate publicly, without the user having to say which
    they have.

    Args:
        state_dict: The checkpoint's parameter mapping.

    Returns:
        ``(architecture, depth)`` where architecture is ``"resnet18_dual"`` or
        ``"resnet_fc"``. ``depth`` is 0 for the dual-head model.

    Raises:
        ValueError: If the checkpoint matches no known layout.
    """
    if "coord_head.weight" in state_dict and "visibility_head.weight" in state_dict:
        return "resnet18_dual", 0

    fc = state_dict.get("fc.weight")
    if fc is not None and fc.shape[0] == NUM_KEYPOINTS * 2:
        # Feature width identifies the backbone: bottleneck ResNets end at 2048.
        depth = {512: 18, 2048: 50}.get(int(fc.shape[1]))
        if depth is None:
            raise ValueError(
                f"fc layer has {fc.shape[1]} input features; expected 512 (ResNet18/34) "
                f"or 2048 (ResNet50)"
            )
        # ResNet18 and ResNet34 share a feature width; block count separates them.
        if depth == 18 and any(k.startswith("layer1.2.") for k in state_dict):
            depth = 34
        return "resnet_fc", depth

    raise ValueError(
        "unrecognised court keypoint checkpoint: expected either a dual-head model "
        "(coord_head/visibility_head) or a ResNet with fc -> "
        f"{NUM_KEYPOINTS * 2} outputs"
    )


class CourtKeypointDetector:
    """Predicts the 14 court keypoints in a frame.

    Supports this project's dual-head ResNet18 and single-head ResNet regressors, and
    reorders the model's outputs into this project's canonical keypoint order.
    """

    def __init__(self, config: CourtConfig, device: torch.device | None = None) -> None:
        """Load the trained keypoint model.

        Args:
            config: Court model settings.
            device: Torch device. Defaults to the best available.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
            ValueError: If the checkpoint architecture is unrecognised, or
                ``config.keypoint_order`` is not a valid permutation.
        """
        self.config = config
        self.device = device or select_device()

        path = Path(config.model)
        if not path.exists():
            raise FileNotFoundError(
                f"court keypoint weights not found at {path}. "
                f"Train them with scripts/train_court_keypoints.py (see README)."
            )

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        architecture, depth = identify_architecture(state_dict)
        if config.architecture != "auto" and config.architecture != architecture:
            logger.warning(
                "config says court.architecture=%s but the checkpoint looks like %s; "
                "trusting the checkpoint",
                config.architecture,
                architecture,
            )
        self.architecture = architecture

        if architecture == "resnet18_dual":
            self.model = CourtKeypointNet(pretrained=False)
        else:
            self.model = build_resnet_fc_regressor(depth)
            logger.info("loaded single-head ResNet%d court model", depth)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self._order = _validate_order(config.keypoint_order)
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict court keypoints for one frame.

        Args:
            frame: BGR image.

        Returns:
            ``(keypoints, confidences)`` — ``(14, 2)`` pixel coordinates in the
            *original* frame's scale, in canonical keypoint order, and ``(14,)``
            confidences. Single-head models have no visibility branch, so they report
            full confidence throughout.
        """
        import cv2

        height, width = frame.shape[:2]
        size = self.config.input_size
        resized = cv2.resize(frame, (size, size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self._mean) / self._std
        tensor = tensor.unsqueeze(0).to(self.device)

        if self.architecture == "resnet18_dual":
            coords, visibility_logits = self.model(tensor)
            # Normalised [0, 1] -> original frame pixels.
            keypoints = coords[0].cpu().numpy() * np.array([width, height])
            confidences = torch.sigmoid(visibility_logits)[0].cpu().numpy()
        else:
            # Single-head models emit raw pixels in the network's input resolution.
            raw = self.model(tensor)[0].cpu().numpy().reshape(NUM_KEYPOINTS, 2)
            keypoints = raw * np.array([width / size, height / size])
            confidences = np.ones(NUM_KEYPOINTS)

        if self._order is not None:
            keypoints = keypoints[self._order]
            confidences = confidences[self._order]

        return keypoints.astype(np.float64), confidences.astype(np.float64)


def _validate_order(order: list[int] | None) -> np.ndarray | None:
    """Validate a keypoint permutation, returning it as an index array.

    Args:
        order: Permutation mapping canonical index to checkpoint output index.

    Returns:
        The permutation as an array, or ``None`` if no reordering is needed.

    Raises:
        ValueError: If ``order`` is not a permutation of ``0..NUM_KEYPOINTS-1``.
    """
    if order is None:
        return None
    if sorted(order) != list(range(NUM_KEYPOINTS)):
        raise ValueError(
            f"court.keypoint_order must be a permutation of 0..{NUM_KEYPOINTS - 1}, "
            f"got {order}"
        )
    return np.asarray(order, dtype=int)


class HomographyTracker:
    """Maintains a court homography across frames, refitting periodically.

    For fixed-camera match footage the court is near-static, so refitting every frame
    is wasted compute. This refits every ``refit_interval`` frames and reuses the last
    accepted homography in between. A refit that fails or reprojects poorly is
    rejected, leaving the previous good homography in place — a brief occlusion of the
    baseline should not knock out speed measurement for the whole rally.
    """

    def __init__(self, config: CourtConfig) -> None:
        """Initialise the tracker.

        Args:
            config: Court settings, including quality gates and refit interval.
        """
        self.config = config
        self._homography: CourtHomography | None = None
        self._keypoints: np.ndarray | None = None
        self._frames_since_fit = 0
        self._rejections = 0

    @property
    def homography(self) -> CourtHomography | None:
        """The most recently accepted homography, if any."""
        return self._homography

    @property
    def keypoints(self) -> np.ndarray | None:
        """The keypoints behind the most recently accepted homography."""
        return self._keypoints

    @property
    def rejection_count(self) -> int:
        """How many candidate fits have been rejected so far."""
        return self._rejections

    def should_refit(self) -> bool:
        """Whether this frame is due for a keypoint detection + refit."""
        return self._homography is None or self._frames_since_fit >= self.config.refit_interval

    def offer(self, keypoints: np.ndarray, confidences: np.ndarray) -> bool:
        """Offer a fresh keypoint detection as a homography candidate.

        Args:
            keypoints: ``(14, 2)`` detected pixel coordinates.
            confidences: ``(14,)`` per-keypoint confidences.

        Returns:
            ``True`` if the candidate was accepted and is now the active homography.
        """
        self._frames_since_fit = 0
        try:
            candidate = CourtHomography.from_keypoints(
                keypoints,
                confidences,
                min_confidence=self.config.min_keypoint_confidence,
                ransac_threshold=self.config.ransac_threshold,
            )
        except HomographyError as exc:
            self._rejections += 1
            logger.debug("homography fit failed: %s", exc)
            return False

        if candidate.reprojection_error > self.config.max_reprojection_error_m:
            self._rejections += 1
            logger.debug(
                "homography rejected: reprojection error %.3f m > %.3f m",
                candidate.reprojection_error,
                self.config.max_reprojection_error_m,
            )
            return False

        self._homography = candidate
        self._keypoints = np.asarray(keypoints, dtype=np.float64)
        return True

    def tick(self) -> None:
        """Advance the refit counter by one frame."""
        self._frames_since_fit += 1


def validate_keypoint_array(keypoints: np.ndarray) -> np.ndarray:
    """Validate and normalise a keypoint array to ``(14, 2)`` float64.

    Args:
        keypoints: Array-like of keypoint coordinates.

    Returns:
        The validated ``(14, 2)`` float64 array.

    Raises:
        ValueError: If the shape is not ``(14, 2)``.
    """
    array = np.asarray(keypoints, dtype=np.float64)
    if array.shape != (NUM_KEYPOINTS, 2):
        raise ValueError(f"expected ({NUM_KEYPOINTS}, 2) keypoints, got {array.shape}")
    return array
