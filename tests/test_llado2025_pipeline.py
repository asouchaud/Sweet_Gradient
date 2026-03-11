from __future__ import annotations

import pytest
import torch

from scripts.perceptual_models.llado2025.llado2025_distance import llado2025_distance
from scripts.perceptual_models.llado2025.llado2025_estimatelocalisation import (
    llado2025_estimatelocalisation_torch,
)


def make_template() -> dict[str, torch.Tensor]:
    # As in the distance tests, we use a simple synthetic three-angle template with 24 bands.
    # The cue values are -1, 0 and +1, corresponding to -30, 0 and +30 degrees.
    angles = torch.tensor([-30.0, 0.0, 30.0], dtype=torch.float32).view(-1, 1)
    cues = (angles / 30.0).repeat(1, 24)

    return {
        "itd": cues.clone(),
        "ild": cues.clone(),
        "azimuths": angles.clone(),
    }


def estimate_localisation(target_value: float) -> torch.Tensor:
    # This helper runs the same distance + localisation stages as the actual model, but on a small
    # synthetic cue set where the expected output is easy to reason about.
    template = make_template()
    target = {
        "itd": torch.full((1, 1, 24), target_value, dtype=torch.float32),
        "ild": torch.full((1, 1, 24), target_value, dtype=torch.float32),
    }

    distance = llado2025_distance(target, template)
    return llado2025_estimatelocalisation_torch(distance, template)


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