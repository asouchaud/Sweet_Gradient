from __future__ import annotations

import pytest
import torch

from scripts.perceptual_models.desena2020.desena2020_estimateuncertainty import (
    desena2020_estimateuncertainty,
)


def test_desena2020_uncertainty_is_zero_for_single_certain_direction_when_hmin_is_zero() -> None:
    # A one-hot likelihood at 0 degrees produces a unit resultant vector, so H = 0 and the
    # normalized uncertainty should also be 0 when Hmin is 0.
    angles = torch.tensor([-45.0, 0.0, 45.0, 90.0], dtype=torch.float64).view(-1, 1)
    distance = {
        "weightedLikelihood": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=torch.float64)
    }

    uncertainty = desena2020_estimateuncertainty(distance, angles, Hmin=0.0)

    # assert uncertainty.shape == (1, 1, 1)
    assert uncertainty.shape == (1, 1)
    assert uncertainty.item() == pytest.approx(0.0, abs=1e-8)


def test_desena2020_uncertainty_is_one_for_uniform_likelihood_when_resultant_vector_cancels_out() -> None:
    # With doubled angles at -90, 0, 90 and 180 degrees, a uniform likelihood makes the cosine and
    # sine resultants cancel to zero, which yields the maximum possible uncertainty H = 1.
    angles = torch.tensor([-45.0, 0.0, 45.0, 90.0], dtype=torch.float64).view(-1, 1)
    distance = {
        "weightedLikelihood": torch.full((1, 1, 4), 0.25, dtype=torch.float64)
    }

    uncertainty = desena2020_estimateuncertainty(distance, angles, Hmin=0.0)

    # assert uncertainty.shape == (1, 1, 1)
    assert uncertainty.shape == (1, 1)
    assert uncertainty.item() == pytest.approx(1.0, abs=1e-8)