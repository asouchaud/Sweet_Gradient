from __future__ import annotations

import pytest
import torch

from scripts.optimisation.regularisation import (
    first_moment_regularisation,
    rms_normalisation,
    second_moment_regularisation,
    select_regularistion,
)


def test_select_regularisation_returns_expected_types() -> None:
    # This checks the small dispatch helper that selects the project's regularisation object from the
    # integer flag used in the training code.
    assert select_regularistion(0, n_samples=8, device="cpu") is None
    assert isinstance(select_regularistion(1, n_samples=8, device="cpu"), first_moment_regularisation)
    assert isinstance(select_regularistion(2, n_samples=8, device="cpu"), second_moment_regularisation)

    # Any unknown flag should fall back to None according to the implementation.
    assert select_regularistion(99, n_samples=8, device="cpu") is None


def test_rms_normalisation_gives_unit_rms_per_filter() -> None:
    # This validates the project's RMS normaliser. The expected output is not a hard-coded tensor;
    # instead, the invariant is that every filter should end up with RMS equal to 1.
    filters = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]], dtype=torch.float32)

    normalized = rms_normalisation(filters)
    rms = torch.sqrt(torch.mean(normalized ** 2, dim=-1))

    # Because the helper divides each filter by its own RMS, the resulting RMS must be exactly 1
    # up to floating-point tolerance.
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-6)


def test_first_moment_regularisation_matches_impulse_location() -> None:
    # The first-moment regulariser computes the energy-weighted sample index.
    # For a unit impulse located at index 3, all energy is concentrated at that position, so the
    # energy-weighted mean index must be exactly 3.
    reg = first_moment_regularisation(n_samples=5, device="cpu", dtype=torch.float32)

    impulse = torch.zeros(1, 1, 5, dtype=torch.float32)
    impulse[..., 3] = 1.0

    assert reg.compute_loss(impulse).item() == pytest.approx(3.0)


def test_second_moment_regularisation_is_zero_for_impulse_and_positive_for_spread() -> None:
    # The second-moment regulariser computes the energy-weighted variance around the first moment.
    reg = second_moment_regularisation(n_samples=5, device="cpu", dtype=torch.float32)

    impulse = torch.zeros(1, 1, 5, dtype=torch.float32)
    impulse[..., 2] = 1.0

    spread = torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    # For the impulse, all mass is at the mean index 2, so the variance is exactly 0.
    assert reg.compute_loss(impulse).item() == pytest.approx(0.0, abs=1e-6)

    # For the spread case, equal energy sits at indices 1 and 3.
    # The mean is therefore (1 + 3) / 2 = 2, and the variance is
    # ((1-2)^2 + (3-2)^2) / 2 = (1 + 1) / 2 = 1.
    assert reg.compute_loss(spread).item() == pytest.approx(1.0, abs=1e-6)