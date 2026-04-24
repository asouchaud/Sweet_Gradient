from __future__ import annotations

from types import SimpleNamespace

import pytest
from pathlib import Path
import torch

from playground.ICLD_ICTD.scripts import propagation
from scripts.helpers.miscelaneous import sofa_find

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

def test_single_source_free_field_propagation_computes_silent_output_for_silent_input(monkeypatch) -> None:
    monkeypatch.setattr(free_field_propagation.sf, "read_sofa", lambda _path: _mock_sofa())

    stim = torch.zeros((1, 1, 480), dtype=torch.float64)
    spk_filters = torch.tensor([[[1.0]]], dtype=torch.float64)
    lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    lstn_idx = torch.tensor([0], dtype=torch.long)
    spk_pos = torch.tensor([[2.0, 0.0]], dtype=torch.float64)

    propagation = free_field_propagation.SingleSourceBinauralFreeFieldPropagation(
        fs=48000,
        lstn_pos=lstn_pos,
        source_pos=spk_pos[0],
        sofa_file_path="unused.sofa",
    )

    output = propagation(stim, spk_filters, lstn_idx)

    expected = torch.zeros((1, 1, 2, 1), dtype=torch.float64)

    assert torch.allclose(output, expected)


def test_single_source_free_field_propagation_computes_expected_hrirs_for_unit_impulse_input(monkeypatch) -> None:

    repo_root = Path(__file__).resolve().parent.parent

    spk_filters = torch.tensor([[[1.0]]], dtype=torch.float64)
    spk_pos = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    lstn_idx = torch.tensor([0], dtype=torch.long)
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    
    propagation_0 = free_field_propagation.SingleSourceBinauralFreeFieldPropagation(
        fs=48000,
        lstn_pos=lstn_pos,
        source_pos=spk_pos[0],
        sofa_file_path=sofa_path,
    )

    propagation_90 = free_field_propagation.SingleSourceBinauralFreeFieldPropagation(
        fs=48000,
        lstn_pos=lstn_pos,
        source_pos=spk_pos[1],
        sofa_file_path=sofa_path,
    )

    propagation_180 = free_field_propagation.SingleSourceBinauralFreeFieldPropagation(
        fs=48000,
        lstn_pos=lstn_pos,
        source_pos=spk_pos[2],
        sofa_file_path=sofa_path,
    )

    propagation_270 = free_field_propagation.SingleSourceBinauralFreeFieldPropagation(
        fs=48000,
        lstn_pos=lstn_pos,
        source_pos=spk_pos[3],
        sofa_file_path=sofa_path,
    )

    idx_0 = sofa_find(0, propagation_0.sofaile)
    expected_hrir_0 = torch.tensor(propagation_0.sofaile.Data_IR[idx_0], dtype=torch.float64).unsqueeze(0)

    idx_90 = sofa_find(90, propagation_0.sofaile)
    expected_hrir_90 = torch.tensor(propagation_0.sofaile.Data_IR[idx_90], dtype=torch.float64).unsqueeze(0)

    idx_180 = sofa_find(180, propagation_0.sofaile)
    expected_hrir_180 = torch.tensor(propagation_0.sofaile.Data_IR[idx_180], dtype=torch.float64).unsqueeze(0)

    idx_270 = sofa_find(270, propagation_0.sofaile)
    expected_hrir_270 = torch.tensor(propagation_0.sofaile.Data_IR[idx_270], dtype=torch.float64).unsqueeze(0)

    stim = torch.zeros((1,1,expected_hrir_0.shape[-1]), dtype=torch.float64)
    stim[0, 0, 0] = 1.0

    output_0 = propagation_0(stim, spk_filters, lstn_idx)
    output_90 = propagation_90(stim, spk_filters, lstn_idx)
    output_180 = propagation_180(stim, spk_filters, lstn_idx)
    output_270 = propagation_270(stim, spk_filters, lstn_idx)

    assert torch.allclose(output_0, expected_hrir_0)
    assert torch.allclose(output_90, expected_hrir_90)
    assert torch.allclose(output_180, expected_hrir_180)
    assert torch.allclose(output_270, expected_hrir_270)


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
