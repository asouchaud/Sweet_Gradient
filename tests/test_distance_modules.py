from __future__ import annotations

import pytest
import torch

from scripts.helpers.template_comparison import itd_ild_template_distance


def make_template() -> dict[str, torch.Tensor]:
    # We build a tiny synthetic template with 3 azimuth candidates and 24 frequency bands.
    # The ITD and ILD templates are identical here so that the best match is unambiguous.
    angles = torch.tensor([-30.0, 0.0, 30.0], dtype=torch.float32).view(-1, 1)
    cues = (angles / 30.0).repeat(1, 24)

    return {
        "itd": cues.clone(),
        "ild": cues.clone(),
        "azimuths": angles.clone(),
    }


@pytest.mark.parametrize("distance_fn", [itd_ild_template_distance, itd_ild_template_distance])
def test_distance_module_prefers_exact_template_match(distance_fn) -> None:
    # This is a larger behavioural test for both distance modules. We choose a target whose ITD and
    # ILD are all +1 across the 24 bands. In make_template(), the +30 degree entry also has cue
    # value +1 in every band, so it should be the exact match after normalisation.
    template = make_template()
    target = {
        "itd": torch.ones((1, 1, 24), dtype=torch.float32),
        "ild": torch.ones((1, 1, 24), dtype=torch.float32),
    }

    distance = distance_fn(target, template)

    # The implementation compares one batch item, one listener position, three template angles and
    # 24 frequency bands, so those are the expected tensor shapes.
    assert distance["combined"].shape == (1, 1, 3, 24)
    assert distance["likelihood_f"].shape == (1, 1, 3, 24)

    best_match = distance["likelihood_f"].argmax(dim=2)

    # Index 2 corresponds to the third template entry, i.e. +30 degrees. Because the target matches
    # that entry exactly in every band, the highest likelihood should be at index 2 for all 24 bands.
    expected = torch.full((1, 1, 24), 2, dtype=torch.long)

    assert torch.equal(best_match, expected)