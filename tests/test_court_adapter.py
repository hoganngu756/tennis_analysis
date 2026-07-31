"""Tests for third-party court-checkpoint support and mini-court placement.

Two things here have burned this project already and are cheap to pin down:

* **Architecture fingerprinting** — checkpoints arrive in more than one layout, and
  guessing wrong produces a load error at best.
* **Keypoint ordering** — a wrong permutation does *not* raise. RANSAC still fits a
  plausible homography and every downstream distance is silently wrong, so the
  permutation validator is the only thing standing between a typo and bad data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tennis_analysis.config import MiniCourtConfig
from tennis_analysis.court_keypoints.detector import _validate_order, identify_architecture
from tennis_analysis.court_keypoints.geometry import (
    COURT_KEYPOINTS_M,
    COURT_LENGTH,
    DOUBLES_WIDTH,
    NUM_KEYPOINTS,
)
from tennis_analysis.court_keypoints.mini_court import MiniCourt
from tennis_analysis.court_keypoints.model import CourtKeypointNet, build_resnet_fc_regressor


def fc_state_dict(features: int, blocks_in_layer1: int = 2) -> dict[str, torch.Tensor]:
    """Build a minimal single-head ResNet-style state dict for fingerprinting."""
    state = {
        "conv1.weight": torch.zeros(64, 3, 7, 7),
        "fc.weight": torch.zeros(NUM_KEYPOINTS * 2, features),
        "fc.bias": torch.zeros(NUM_KEYPOINTS * 2),
    }
    for block in range(blocks_in_layer1):
        state[f"layer1.{block}.conv1.weight"] = torch.zeros(1)
    return state


class TestIdentifyArchitecture:
    """Fingerprinting a checkpoint from its parameter names and shapes."""

    def test_recognises_this_projects_dual_head_model(self):
        state = CourtKeypointNet(pretrained=False).state_dict()
        assert identify_architecture(state) == ("resnet18_dual", 0)

    def test_recognises_resnet50_single_head(self):
        assert identify_architecture(fc_state_dict(2048)) == ("resnet_fc", 50)

    def test_recognises_resnet18_single_head(self):
        assert identify_architecture(fc_state_dict(512, blocks_in_layer1=2)) == (
            "resnet_fc",
            18,
        )

    def test_distinguishes_resnet34_by_block_count(self):
        """ResNet18 and 34 share a 512-wide fc; only depth of layer1 separates them."""
        assert identify_architecture(fc_state_dict(512, blocks_in_layer1=3)) == (
            "resnet_fc",
            34,
        )

    def test_real_resnet50_regressor_round_trips(self):
        state = build_resnet_fc_regressor(50).state_dict()
        assert identify_architecture(state) == ("resnet_fc", 50)

    def test_unknown_feature_width_raises(self):
        with pytest.raises(ValueError, match="expected 512"):
            identify_architecture(fc_state_dict(1024))

    def test_wrong_output_count_is_not_recognised(self):
        state = fc_state_dict(2048)
        state["fc.weight"] = torch.zeros(1000, 2048)  # an ImageNet classifier
        with pytest.raises(ValueError, match="unrecognised"):
            identify_architecture(state)

    def test_empty_state_dict_raises(self):
        with pytest.raises(ValueError, match="unrecognised"):
            identify_architecture({})


class TestBuildResnetFcRegressor:
    """The single-head architecture builder."""

    @pytest.mark.parametrize("depth", [18, 34, 50])
    def test_output_shape_is_two_per_keypoint(self, depth):
        model = build_resnet_fc_regressor(depth)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 224, 224))
        assert out.shape == (1, NUM_KEYPOINTS * 2)

    def test_unsupported_depth_raises(self):
        with pytest.raises(ValueError, match="unsupported ResNet depth"):
            build_resnet_fc_regressor(101)


class TestValidateOrder:
    """The keypoint permutation guard."""

    def test_none_means_no_reordering(self):
        assert _validate_order(None) is None

    def test_identity_permutation_is_accepted(self):
        order = _validate_order(list(range(NUM_KEYPOINTS)))
        np.testing.assert_array_equal(order, np.arange(NUM_KEYPOINTS))

    def test_real_world_swap_is_accepted(self):
        """The 5<->6 singles-corner swap seen in public checkpoints."""
        order = _validate_order([0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12, 13])
        assert order[5] == 6 and order[6] == 5

    def test_permutation_actually_reorders(self):
        order = _validate_order([0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12, 13])
        source = np.arange(NUM_KEYPOINTS * 2).reshape(NUM_KEYPOINTS, 2)
        reordered = source[order]
        np.testing.assert_array_equal(reordered[5], source[6])
        np.testing.assert_array_equal(reordered[6], source[5])
        np.testing.assert_array_equal(reordered[0], source[0])

    def test_duplicate_index_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            _validate_order([0] * NUM_KEYPOINTS)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            _validate_order([0, 1, 2])

    def test_out_of_range_index_raises(self):
        bad = list(range(NUM_KEYPOINTS - 1)) + [99]
        with pytest.raises(ValueError, match="permutation"):
            _validate_order(bad)


class TestMiniCourtPlacement:
    """Canvas mapping, including the run-off surround around the court."""

    @pytest.fixture
    def mini(self) -> MiniCourt:
        return MiniCourt(MiniCourtConfig(width=220, height=370, surround_m=4.0))

    def test_court_corners_are_on_canvas(self, mini):
        for point in COURT_KEYPOINTS_M:
            assert mini.is_on_canvas(tuple(point))

    def test_court_centre_maps_near_canvas_centre(self, mini):
        px, py = mini.court_to_canvas((DOUBLES_WIDTH / 2, COURT_LENGTH / 2))
        assert px == pytest.approx(220 / 2, abs=3)
        assert py == pytest.approx(370 / 2, abs=3)

    def test_player_behind_the_baseline_still_renders(self, mini):
        """The bug this surround exists to fix: receivers stand behind the baseline."""
        assert mini.is_on_canvas((5.0, COURT_LENGTH + 3.8))
        assert mini.is_on_canvas((5.0, -3.3))

    def test_point_beyond_the_surround_is_off_canvas(self, mini):
        assert not mini.is_on_canvas((5.0, COURT_LENGTH + 40.0))
        assert not mini.is_on_canvas((-50.0, 12.0))

    def test_zero_surround_clips_behind_the_baseline(self):
        tight = MiniCourt(MiniCourtConfig(width=220, height=370, surround_m=0.0))
        assert not tight.is_on_canvas((5.0, COURT_LENGTH + 3.8))

    def test_ordering_is_preserved_along_each_axis(self, mini):
        near = mini.court_to_canvas((5.0, 2.0))
        far = mini.court_to_canvas((5.0, 20.0))
        left = mini.court_to_canvas((1.0, 12.0))
        right = mini.court_to_canvas((9.0, 12.0))
        assert near[1] < far[1]
        assert left[0] < right[0]

    def test_render_places_both_players_and_ball(self, mini):
        canvas = mini.render(
            {1: (5.0, COURT_LENGTH + 2.0), 3: (6.0, -2.0)}, ball=(5.5, 12.0)
        )
        assert canvas.shape == (370, 220, 3)
        # Markers are drawn in colours absent from the flat background surface.
        assert len(np.unique(canvas.reshape(-1, 3), axis=0)) > 3

    def test_render_survives_offscreen_positions(self, mini):
        canvas = mini.render({1: (5.0, 500.0)}, ball=(-90.0, 3.0))
        assert canvas.shape == (370, 220, 3)

    def test_render_ignores_nan_positions(self, mini):
        canvas = mini.render({1: (np.nan, np.nan)}, ball=None)
        assert canvas.shape == (370, 220, 3)
