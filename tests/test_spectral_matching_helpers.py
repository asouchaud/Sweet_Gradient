from __future__ import annotations

import pytest
import torch

from scripts.perceptual_models.spectral_matching.DEPRACATED_spectral_matching_helpers import (
    log_spectral_distortion,
)


def test_log_spectral_distortion_is_zero_for_identical_signals() -> None:
    # Comparing a signal against itself should give exactly zero spectral distortion.
    x = torch.tensor(
        [[[1.0, 0.0, -1.0, 0.5], [0.5, -0.5, 0.25, -0.25]]],
        dtype=torch.float32,
    )

    lsd = log_spectral_distortion(x, x)

    assert lsd.shape == (1,)
    assert lsd.item() == pytest.approx(0.0, abs=1e-6)


def test_log_spectral_distortion_is_symmetric_and_detects_level_changes() -> None:
    # LSD is based on squared log-magnitude differences, so it should be symmetric in its inputs.
    # The unchanged example should stay at zero, while a level-scaled example should become
    # strictly positive.
    x = torch.tensor(
        [
            [[1.0, 0.0, -1.0, 0.5], [0.5, -0.5, 0.25, -0.25]],
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    y = x.clone()
    y[1] = 0.5 * y[1]

    lsd_xy = log_spectral_distortion(x, y, n_fft=8)
    lsd_yx = log_spectral_distortion(y, x, n_fft=8)

    assert lsd_xy.shape == (2,)
    assert lsd_xy[0].item() == pytest.approx(0.0, abs=1e-6)
    assert lsd_xy[1].item() > 0.0
    assert torch.allclose(lsd_xy, lsd_yx, atol=1e-6)