from __future__ import annotations

import pytest
import torch

from pathlib import Path

from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching


def make_model(lstn_pos=None) -> spectral_matching:
    repo_root = Path(__file__).resolve().parent.parent

    dummy_loss = torch.nn.functional.mse_loss
    dummy_target = 0
    dummy_target_pos = torch.tensor([0.0, 1.0], dtype=torch.float64)
    dummy_fs = 48000
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64) if lstn_pos is None else lstn_pos
    spectral_model = spectral_matching(dummy_loss, dummy_target, dummy_target_pos, dummy_fs, dummy_sofa_path, dummy_lstn_pos)

    return spectral_model


def test_log_spectral_distortion_is_zero_for_identical_signals() -> None:
    # Comparing a signal against itself should give exactly zero spectral distortion.
    
    model = make_model()
    x = torch.tensor(
        [[[1.0, 0.0, -1.0, 0.5], [0.5, -0.5, 0.25, -0.25]]],
        dtype=torch.float32,
    )

    lsd = model._log_spectral_distortion(x, x)

    assert lsd.shape == (1,)
    assert lsd.item() == pytest.approx(0.0, abs=1e-6)


def test_log_spectral_distortion_is_symmetric_and_detects_level_changes() -> None:
    # LSD is based on squared log-magnitude differences, so it should be symmetric in its inputs.
    # The unchanged example should stay at zero, while a level-scaled example should become
    # strictly positive.
    model = make_model()
    model.n_fft = 8  
    x = torch.tensor(
        [
            [[1.0, 0.0, -1.0, 0.5], [0.5, -0.5, 0.25, -0.25]],
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    y = x.clone()
    y[1] = 0.5 * y[1]

    lsd_xy = model._log_spectral_distortion(x, y)
    lsd_yx = model._log_spectral_distortion(y, x)

    assert lsd_xy.shape == (2,)
    assert lsd_xy[0].item() == pytest.approx(0.0, abs=1e-6)
    assert lsd_xy[1].item() > 0.0
    assert torch.allclose(lsd_xy, lsd_yx, atol=1e-6)


def test_spectral_matching_output_shape() -> None:
    # The forward method should return a tensor of shape (batch, n_listeners).
    lstn_pos = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    model = make_model(lstn_pos=lstn_pos)
    
    batch_size = 4
    n_listeners = len(lstn_pos)
    
    bin_stim = torch.randn(batch_size, n_listeners, 2, 16)
    reference_stim = torch.randn(batch_size, 1, 16, dtype=torch.float64)
    listener_indices = torch.arange(n_listeners)

    output = model.forward(bin_stim, reference_stim, listener_indices)

    assert output.shape == (batch_size, n_listeners)