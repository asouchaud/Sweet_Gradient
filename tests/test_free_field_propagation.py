from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sofar")

from scripts.propagation import free_field_propagation


def _mock_sofa() -> SimpleNamespace:
    # A single SOFA entry is enough for these tests because we only care about distance-dependent
    # gain, not direction-dependent HRIR selection. The HRIR is a unit impulse in both ears so the
    # propagated output mirrors the applied path gain exactly.
    return SimpleNamespace(
        SourcePosition=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).numpy(),
        Data_IR=torch.tensor([[[1.0], [1.0]]], dtype=torch.float64).numpy(),
    )


def test_binaural_free_field_propagation_optionally_applies_inverse_distance_gain(monkeypatch) -> None:
    monkeypatch.setattr(free_field_propagation.sf, "read_sofa", lambda _path: _mock_sofa())

    stim = torch.tensor([[[1.0]]], dtype=torch.float64)
    spk_filters = torch.tensor([[[1.0]]], dtype=torch.float64)
    lstn_pos = [[0.0, 0.0], [1.0, 0.0]]
    lstn_idx = [0, 1]  
    spk_pos = [[2.0, 0.0]]

    propagation_without_attenuation = free_field_propagation.BinauralFreeFieldPropagation(
        fs=48000,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        sofa_file_path="unused.sofa",
    )
    propagation_with_attenuation = free_field_propagation.BinauralFreeFieldPropagation(
        fs=48000,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        sofa_file_path="unused.sofa",
        apply_distance_attenuation=True,
    )

    output_without_attenuation = propagation_without_attenuation(stim, spk_filters, lstn_idx)
    output_with_attenuation = propagation_with_attenuation(stim, spk_filters, lstn_idx)

    expected_without_attenuation = torch.tensor(
        [[[[1.0], [1.0]], [[1.0], [1.0]]]],
        dtype=torch.float64,
    )
    expected_with_attenuation = torch.tensor(
        [[[[0.5], [0.5]], [[1.0], [1.0]]]],
        dtype=torch.float64,
    )

    assert torch.allclose(output_without_attenuation, expected_without_attenuation)
    assert torch.allclose(output_with_attenuation, expected_with_attenuation)
