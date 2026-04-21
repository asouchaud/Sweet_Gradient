from __future__ import annotations

import pytest

pytest.importorskip("torch.utils.tensorboard")

from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.perceptual_models.ipd.ipd import interaural_phase_difference_matching


def test_ipd_loss_norm_matches_l1() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    dummy_target_pos = [0.0, 1.0]
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    model = interaural_phase_difference_matching(
        loss=F.l1_loss,
        target=0.0,
        target_pos=dummy_target_pos,
        sofa_path=dummy_sofa_path,
        lstn_pos=dummy_lstn_pos,
        fs=48000,
        n_fft=32,
    )
    assert float(model.loss_norm) == pytest.approx(float(torch.pi), rel=0.0, abs=1e-6)


def test_ipd_loss_norm_matches_mse() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    dummy_target_pos = [0.0, 1.0]
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
        
    model = interaural_phase_difference_matching(
        loss=F.mse_loss,
        target=0.0,
        target_pos=dummy_target_pos,
        sofa_path=dummy_sofa_path,
        lstn_pos=dummy_lstn_pos,
        fs=48000,
        n_fft=32,
    )
    assert float(model.loss_norm) == pytest.approx(float(torch.pi) ** 2, rel=0.0, abs=1e-6)


def test_ipd_zero_when_bin_equals_reference() -> None:
    
    repo_root = Path(__file__).resolve().parent.parent

    dummy_target_pos = [0.0, 1.0]
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    dummy_lstn_idx = torch.tensor([0], dtype=torch.long)
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
        

    model = interaural_phase_difference_matching(
        loss=F.l1_loss,
        target=0.0,
        target_pos=dummy_target_pos,
        sofa_path=dummy_sofa_path,
        lstn_pos=dummy_lstn_pos,
        fs=48000,
        n_fft=64,
    )

    out = model(torch.zeros(1,1,2,64, dtype=torch.float64), 
                torch.zeros(1,1,64, dtype=torch.float64), 
                dummy_lstn_idx)
    assert out.shape == (1 * 1,)
    assert float(out.item()) == pytest.approx(0.0, abs=1e-8)


def test_ipd_output_shape_and_bounds() -> None:
    torch.manual_seed(0)

    repo_root = Path(__file__).resolve().parent.parent

    dummy_target_pos = [0.0, 1.0]
    dummy_lstn_pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    batch, n_listeners, t = 2, 4, 128
    dummy_lstn_idx = torch.arange(n_listeners, dtype=torch.long)

    model = interaural_phase_difference_matching(
        loss=F.l1_loss,
        target=0.0,
        target_pos=dummy_target_pos,
        sofa_path=dummy_sofa_path,
        lstn_pos=dummy_lstn_pos,
        fs=48000,
        n_fft=256,
        fmin=200.0,
        fmax=7000.0,
    )

    bin_stim = torch.randn(batch, n_listeners, 2, t, dtype=torch.float64)
    ref = torch.randn(batch, 1, t, dtype=torch.float64)
    out = model(bin_stim, ref, dummy_lstn_idx)

    assert out.shape == (batch * n_listeners,)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= float(torch.pi) + 1e-5)


def test_ipd_mask_empty_does_not_crash() -> None:
    # If the frequency mask is empty, the code should still run and return a valid score.
    repo_root = Path(__file__).resolve().parent.parent

    dummy_target_pos = [0.0, 1.0]
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    dummy_lstn_idx = torch.tensor([0], dtype=torch.long)
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    batch, n_listeners, t = 2, 4, 128
    
    model = interaural_phase_difference_matching(
        loss=F.l1_loss,
        target=0.0,
        target_pos=dummy_target_pos,
        sofa_path=dummy_sofa_path,
        lstn_pos=dummy_lstn_pos,
        fs=48000,
        n_fft=256,
        fmin=8000.0,
        fmax=200.0,
    )
    out = model(torch.zeros(1, 1, 2, 128, dtype=torch.float64),
                torch.zeros(1, 1, 128, dtype=torch.float64),
                dummy_lstn_idx)
    assert out.shape == (1 * 1,)
