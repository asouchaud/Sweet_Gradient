from __future__ import annotations

import pytest
import torch

from scripts.helpers.dsp import causal_conv1d
from scripts.helpers.dsp import l3_loss


def test_causal_conv1d_matches_expected_causal_response() -> None:
    # This checks the project-specific causal convolution helper rather than torch.conv1d itself.
    # The kernel [2, -1] is flipped inside true_conv1d, and the input is left-padded to make the
    # operation causal while preserving the original signal length.
    input_signal = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    kernel = torch.tensor([[[2.0, -1.0]]], dtype=torch.float32)

    output = causal_conv1d(input_signal, kernel)

    # Expected values are computed by hand for a causal FIR:
    # y[0] = 2*1
    # y[1] = 2*2 + (-1)*1
    # y[2] = 2*3 + (-1)*2
    expected = torch.tensor([[[2.0, 3.0, 4.0]]], dtype=torch.float32)

    assert torch.allclose(output, expected)


def test_causal_conv1d_matches_expected_causal_response_2() -> None:
    # This is a second test of the same helper with a different kernel and input, to check that the
    # convolution is being performed correctly and not just accidentally correct for one specific case.
    input_signal = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    kernel = torch.tensor([[[1.0, 0.5, 0.25]]], dtype=torch.float32)

    output = causal_conv1d(input_signal, kernel)

    # Expected values are computed by hand for a causal FIR:
    # y[0] = 1*0
    # y[1] = 1*1 + 0.5*0
    # y[2] = 1*0 + 0.5*1 + 0.25*0
    # y[3] = 1*0 + 0.5*0 + 0.25*1
    expected = torch.tensor([[[0.0, 1.0, 0.5, 0.25]]], dtype=torch.float32)

    assert torch.allclose(output, expected)


def test_causal_conv1d_preserves_input_length() -> None:
    # The helper is written to implement a "same-length" causal convolution by padding the input on
    # the left by K-1 samples before calling conv1d. This test checks that contract directly.
    input_signal = torch.arange(1, 6, dtype=torch.float32).view(1, 1, -1)
    kernel = torch.tensor([[[1.0, 0.5, -0.25]]], dtype=torch.float32)

    output = causal_conv1d(input_signal, kernel)

    # The expected length is therefore exactly the input length, regardless of kernel size.
    assert output.shape == input_signal.shape


def test_l3_loss_supports_mean_sum_and_none_reductions() -> None:
    # This validates the custom l3_loss helper in the project. We use small numbers so the expected
    # outputs can be computed exactly by hand.
    input_tensor = torch.tensor([1.0, 3.0], dtype=torch.float32)
    target_tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)

    unreduced = l3_loss(input_tensor, target_tensor, reduction="none")

    # Absolute differences are [1, 2], so the elementwise cubic loss is [1^3, 2^3] = [1, 8].
    assert torch.allclose(unreduced, torch.tensor([1.0, 8.0], dtype=torch.float32))

    # The summed loss is 1 + 8 = 9.
    assert l3_loss(input_tensor, target_tensor, reduction="sum").item() == pytest.approx(9.0)

    # The mean loss is (1 + 8) / 2 = 4.5.
    assert l3_loss(input_tensor, target_tensor, reduction="mean").item() == pytest.approx(4.5)