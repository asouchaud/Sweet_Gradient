from __future__ import annotations

import pytest
import torch

from scripts.perceptual_models.llado2025.llado2025_helpers import (
    near_field_comp_angles,
    near_field_target,
    softmax_st,
)


def test_near_field_target_at_zero_degrees_lies_on_x_axis() -> None:
    # near_field_target() converts an azimuth to the corresponding target position used by this
    # project. At 0 degrees, tan(0) = 0, so the y-coordinate must be 0, while the x-coordinate is
    # hard-coded in the implementation as 1.732.
    target_pos = near_field_target(0.0)

    assert target_pos[0].item() == pytest.approx(1.732, abs=1e-6)
    assert target_pos[1].item() == pytest.approx(0.0, abs=1e-6)


def test_near_field_comp_angles_are_symmetric_around_centre_listener() -> None:
    # This checks the geometry helper for a symmetric listener layout around the y-axis.
    # The target is straight ahead at 0 degrees, and the listeners are placed at y = 0, +0.5, -0.5.
    target_pos = near_field_target(0.0)
    listeners = [[0.0, 0.0], [0.0, 0.5], [0.0, -0.5]]

    angles = near_field_comp_angles(target_pos, listeners)

    # The centre listener should see the source at exactly 0 degrees.
    assert angles[0].item() == pytest.approx(0.0, abs=1e-6)

    # The upper and lower listeners are mirror images, so their angles should have equal magnitude
    # and opposite sign. The one above the centre must point slightly downward (negative angle),
    # while the one below must point slightly upward (positive angle).
    assert angles[1].item() == pytest.approx(-angles[2].item(), abs=1e-6)
    assert 0.0 > angles[1].item()
    assert angles[2].item() > 0.0


def test_softmax_st_returns_hard_one_hot_on_forward_pass() -> None:
    # softmax_st() is a straight-through estimator. In the forward pass it returns a hard one-hot
    # vector at the argmax location, while keeping softmax gradients for backpropagation.
    z = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)

    y = softmax_st(z, dim=-1, tau=0.5)

    # The largest entry of z is the middle element, so the forward output must be [0, 1, 0].
    expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    assert torch.equal(y, expected)