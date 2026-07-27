"""ResNet18-backbone regression model for the 14 court keypoints.

The head emits, per keypoint, a normalised ``(x, y)`` in ``[0, 1]`` image coordinates
plus a visibility logit. The visibility branch matters because broadcast framing often
crops the far baseline out of shot: without it, the model is forced to hallucinate a
position for keypoints it cannot see, and those hallucinations poison the homography.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .geometry import NUM_KEYPOINTS

#: ImageNet normalisation, matching the pretrained ResNet18 backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CourtKeypointNet(nn.Module):
    """ResNet18 regressor predicting 14 court keypoints and their visibility."""

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True) -> None:
        """Build the network.

        Args:
            num_keypoints: Number of keypoints to regress.
            pretrained: Load ImageNet weights into the backbone. Set ``False`` for
                deterministic tests or when loading a fine-tuned checkpoint.
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.coord_head = nn.Linear(in_features, num_keypoints * 2)
        self.visibility_head = nn.Linear(in_features, num_keypoints)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: Normalised image batch of shape ``(B, 3, H, W)``.

        Returns:
            ``(coords, visibility_logits)`` where ``coords`` is ``(B, K, 2)`` in
            normalised ``[0, 1]`` image space and ``visibility_logits`` is ``(B, K)``.
        """
        features = self.backbone(x)
        coords = self.coord_head(features).view(-1, self.num_keypoints, 2)
        # Sigmoid keeps predictions inside the frame, which is where every keypoint
        # lives in a correctly framed shot.
        coords = torch.sigmoid(coords)
        visibility = self.visibility_head(features)
        return coords, visibility


def keypoint_loss(
    pred_coords: torch.Tensor,
    pred_visibility: torch.Tensor,
    target_coords: torch.Tensor,
    target_visibility: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the combined coordinate + visibility loss.

    Coordinate error is only counted for keypoints that are actually visible in the
    ground truth; invisible keypoints have no meaningful target position.

    Args:
        pred_coords: ``(B, K, 2)`` predicted normalised coordinates.
        pred_visibility: ``(B, K)`` predicted visibility logits.
        target_coords: ``(B, K, 2)`` ground-truth normalised coordinates.
        target_visibility: ``(B, K)`` ground-truth visibility in ``{0, 1}``.

    Returns:
        ``(total, coord_loss, visibility_loss)`` as scalar tensors.
    """
    visibility_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_visibility, target_visibility
    )

    mask = target_visibility.unsqueeze(-1)
    visible_count = mask.sum().clamp(min=1.0)
    # Smooth L1 is standard for keypoint regression: less outlier-sensitive than MSE.
    coord_error = nn.functional.smooth_l1_loss(
        pred_coords * mask, target_coords * mask, reduction="sum"
    )
    coord_loss = coord_error / (visible_count * 2)

    return coord_loss + visibility_loss, coord_loss, visibility_loss
