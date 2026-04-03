from __future__ import annotations

import pytest

pytest.importorskip("torch.utils.tensorboard")

import torch
import torch.nn.functional as F

from scripts.perceptual_models.ipd.ipd import interaural_phase_difference


def test_ipd_loss_norm_matches_l1() -> None:
    ref = torch.zeros(2, 16, dtype=torch.float32)
    model = interaural_phase_difference(
        loss=F.l1_loss,
        target=0.0,
        reference_stim=ref,
        fs=48000,
        n_fft=32,
    )
    assert float(model.loss_norm) == pytest.approx(float(torch.pi), rel=0.0, abs=1e-6)


def test_ipd_loss_norm_matches_mse() -> None:
    ref = torch.zeros(2, 16, dtype=torch.float32)
    model = interaural_phase_difference(
        loss=F.mse_loss,
        target=0.0,
        reference_stim=ref,
        fs=48000,
        n_fft=32,
    )
    assert float(model.loss_norm) == pytest.approx(float(torch.pi) ** 2, rel=0.0, abs=1e-6)


def test_ipd_zero_when_bin_equals_reference() -> None:
    ref = torch.zeros(2, 64, dtype=torch.float32)
    model = interaural_phase_difference(
        loss=F.l1_loss,
        target=0.0,
        reference_stim=ref,
        fs=48000,
        n_fft=64,
    )
    out = model(ref.clone().view(1, 1, 2, -1))
    assert out.shape == (1 * 1,)
    assert float(out.item()) == pytest.approx(0.0, abs=1e-8)


def test_ipd_output_shape_and_bounds() -> None:
    torch.manual_seed(0)

    batch, n_listeners, t = 2, 4, 128
    ref = torch.randn(2, t, dtype=torch.float32)
    model = interaural_phase_difference(
        loss=F.l1_loss,
        target=0.0,
        reference_stim=ref,
        fs=48000,
        n_fft=256,
        fmin=200.0,
        fmax=7000.0,
    )

    bin_stim = torch.randn(batch, n_listeners, 2, t, dtype=torch.float32)
    out = model(bin_stim)

    assert out.shape == (batch * n_listeners,)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= float(torch.pi) + 1e-5)


def test_ipd_mask_empty_does_not_crash() -> None:
    # If the frequency mask is empty, the code should still run and return a valid score.
    ref = torch.zeros(2, 128, dtype=torch.float32)
    model = interaural_phase_difference(
        loss=F.l1_loss,
        target=0.0,
        reference_stim=ref,
        fs=48000,
        n_fft=256,
        fmin=8000.0,
        fmax=200.0,
    )
    out = model(torch.zeros(1, 1, 2, 128, dtype=torch.float32))
    assert out.shape == (1 * 1,)
