from __future__ import annotations

import pytest
import torch

from pathlib import Path

from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching


def make_model(lstn_pos=None, erb_weighting: bool = True, normalize_log_spectrum: bool = False, n_fft: int = 4096) -> spectral_matching:
    repo_root = Path(__file__).resolve().parent.parent

    dummy_loss = torch.nn.functional.mse_loss
    dummy_target = 0
    dummy_target_pos = torch.tensor([0.0, 1.0], dtype=torch.float64)
    dummy_fs = 48000
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64) if lstn_pos is None else lstn_pos
    spectral_model = spectral_matching(
        dummy_loss,
        dummy_target,
        dummy_target_pos,
        dummy_fs,
        dummy_sofa_path,
        dummy_lstn_pos,
        n_fft=n_fft,
        erb_weighting=erb_weighting,
        normalize_log_spectrum=normalize_log_spectrum,
    )

    return spectral_model


def _tone_signal(freq_hz: float, fs: int, n_samples: int, amplitude: float = 1.0) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float64) / fs
    return amplitude * torch.sin(2 * torch.pi * freq_hz * t)


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
    model = make_model(n_fft=8, normalize_log_spectrum=False)
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


def test_erb_weighting_disabled_matches_uniform() -> None:
    model = make_model(erb_weighting=False, normalize_log_spectrum=False, n_fft=8)
    x = torch.tensor(
        [
            [[1.0, 0.0, -1.0, 0.5], [0.5, -0.5, 0.25, -0.25]],
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    y = x.clone()
    y[1] = 0.5 * y[1]

    lsd = model._log_spectral_distortion(x, y)
    sq_err = (20 * torch.log10(torch.abs(torch.fft.rfft(x, n=8, dim=-1)) + model.eps)
              - 20 * torch.log10(torch.abs(torch.fft.rfft(y, n=8, dim=-1)) + model.eps)) ** 2
    expected = torch.sqrt(torch.mean(sq_err, dim=-1)).mean(dim=-1)

    assert torch.allclose(lsd, expected, atol=1e-6)


def test_erb_weighting_changes_high_frequency_dominance() -> None:
    model_on = make_model(erb_weighting=True, normalize_log_spectrum=False)
    model_off = make_model(erb_weighting=False, normalize_log_spectrum=False)

    fs = 48000
    n_samples = 4096
    perturbation = 0.5

    x = 0.1 * _tone_signal(1000.0, fs, n_samples).view(1, 1, -1).expand(1, 2, -1)
    y_low = x + perturbation * _tone_signal(117.2, fs, n_samples).view(1, 1, -1).expand(1, 2, -1)
    y_high = x + perturbation * _tone_signal(10605.5, fs, n_samples).view(1, 1, -1).expand(1, 2, -1)

    lsd_on_low = model_on._log_spectral_distortion(x, y_low).item()
    lsd_on_high = model_on._log_spectral_distortion(x, y_high).item()
    lsd_off_low = model_off._log_spectral_distortion(x, y_low).item()
    lsd_off_high = model_off._log_spectral_distortion(x, y_high).item()

    ratio_on = lsd_on_high / lsd_on_low
    ratio_off = lsd_off_high / lsd_off_low

    assert ratio_on < ratio_off