from __future__ import annotations

import pytest
import torch
from pathlib import Path

from scripts.perceptual_models.desena2020.desena2020 import desena2020


def make_model() -> desena2020:
    
    repo_root = Path(__file__).resolve().parent.parent

    dummy_loss = torch.nn.functional.mse_loss
    dummy_target = 0
    dummy_fs = 48000
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    desena2020_model = desena2020(dummy_loss, dummy_target, dummy_fs, dummy_sofa_path)

    return desena2020_model

def test_desena2020_uncertainty_is_zero_for_single_certain_direction_when_hmin_is_zero() -> None:
    # A one-hot likelihood at 0 degrees produces a unit resultant vector, so H = 0 and the
    # normalized uncertainty should also be 0 when Hmin is 0.
    
    desena2020_model = make_model()
    desena2020_model.Hmin = 0
    weighted_likelihood = torch.zeros(1,1,37, dtype=torch.float64)
    weighted_likelihood[0,0,18] = 1.0
    distance = {
        "weightedLikelihood": weighted_likelihood
    }

    uncertainty = desena2020_model._estimate_uncertainty(distance)

    assert uncertainty.shape == (1, 1)
    assert uncertainty.item() == pytest.approx(0.0, abs=1e-8)


def test_desena2020_uncertainty_is_one_for_uniform_likelihood_when_resultant_vector_cancels_out() -> None:
    # With doubled angles at -90, 0, 90 and 180 degrees, a uniform likelihood makes the cosine and
    # sine resultants cancel to zero, which yields the maximum possible uncertainty H = 1.
    desena2020_model = make_model()
    desena2020_model.Hmin = 0
    desena2020_model.template['azimuths'] = torch.tensor([-90, 0 , 90 ,180], dtype=torch.float64).view(-1,1)
    distance = {
        "weightedLikelihood": torch.full((1, 1, 4), 0.25, dtype=torch.float64)
    }

    uncertainty = desena2020_model._estimate_uncertainty(distance)

    assert uncertainty.shape == (1, 1)
    assert uncertainty.item() == pytest.approx(1.0, abs=1e-8)