from __future__ import annotations

import pytest
import torch
import numpy as np 
from pathlib import Path

from scripts.helpers.template_comparison import itd_ild_template_distance
from scripts.perceptual_models.llado2025.llado2025 import llado2025

def make_model() -> llado2025:
    
    repo_root = Path(__file__).resolve().parent.parent

    dummy_loss = torch.nn.functional.mse_loss
    dummy_target = [100, float(np.tan(np.deg2rad(0.0)) * 100)]
    dummy_fs = 48000
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    llado2025_model = llado2025(dummy_loss, dummy_target, dummy_fs, dummy_sofa_path, dummy_lstn_pos)

    return(llado2025_model)

def make_template() -> dict[str, torch.Tensor]:
    # As in the distance tests, we use a simple synthetic three-angle template with 24 bands.
    # The cue values are -1, 0 and +1, corresponding to -30, 0 and +30 degrees.
    angles = torch.tensor([-30.0, 0.0, 30.0], dtype=torch.float64).view(-1, 1)
    cues = (angles / 30.0).repeat(1, 24)

    return {
        "itd": cues.clone(),
        "ild": cues.clone(),
        "azimuths": angles.clone(),
    }


def estimate_localisation(target_value: float) -> torch.Tensor:
    # This helper runs the same distance + localisation stages as the actual model, but on a small
    # synthetic cue set where the expected output is easy to reason about.
    model = make_model()
    template = make_template()
    target = {
        "itd": torch.full((1, 1, 24), target_value, dtype=torch.float64),
        "ild": torch.full((1, 1, 24), target_value, dtype=torch.float64),
    }
    model.template = template

    distance = itd_ild_template_distance(target, template)
    return model._estimate_localisation(distance)


def test_localisation_estimator_returns_middle_azimuth_for_middle_template() -> None:
    # A target cue value of 0 exactly matches the middle template entry, which corresponds to
    # azimuth 0 degrees, so the estimated localisation should be 0.
    estimated = estimate_localisation(0.0)

    assert estimated.shape == (1, 1)
    assert estimated.item() == pytest.approx(0.0, abs=1e-6)


def test_localisation_estimator_changes_sign_with_target_cues() -> None:
    # A target cue value of -1 exactly matches the -30 degree template entry.
    # A target cue value of +1 exactly matches the +30 degree template entry.
    left_estimate = estimate_localisation(-1.0).item()
    right_estimate = estimate_localisation(1.0).item()

    # The signs should therefore flip with the sign of the cue value.
    assert 0.0 > left_estimate
    assert right_estimate > 0.0

    # Because the matches are exact in every frequency band, the straight-through localisation
    # estimate should recover the corresponding template azimuths exactly: -30 and +30 degrees.
    assert left_estimate == pytest.approx(-30.0, abs=1e-6)
    assert right_estimate == pytest.approx(30.0, abs=1e-6)