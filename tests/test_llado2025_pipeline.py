from __future__ import annotations

import pytest
import torch
import numpy as np 
from pathlib import Path

from scripts.helpers.miscelaneous import get_hrirs_for_template
from scripts.helpers.dsp import causal_conv1d
from scripts.helpers.template_comparison import itd_ild_template_distance
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation


@pytest.fixture(autouse=True)
def _llado2025_tests_inference_mode() -> None:
    # Forward uses retain_grad when grad is enabled; these tests evaluate with inputs that do not require grad.
    with torch.inference_mode():
        yield


def make_model() -> llado2025:
    
    repo_root = Path(__file__).resolve().parent.parent

    dummy_loss = torch.nn.functional.mse_loss
    dummy_target = [100, float(np.tan(np.deg2rad(0.0)) * 100)]
    dummy_fs = 48000
    dummy_lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    dummy_sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    llado2025_model = llado2025(dummy_loss, dummy_target, dummy_fs, dummy_sofa_path, dummy_lstn_pos, balance_interaural_cue_grads=False)

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
        "itd": torch.full((1, 1, 24), target_value, dtype=torch.float64, requires_grad=True),
        "ild": torch.full((1, 1, 24), target_value, dtype=torch.float64, requires_grad=True),
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


def test_llado2025_output_shape() -> None:
    # The model should output a single azimuth estimate per listener.
    model = make_model()
    
    batch_size = 4
    n_listeners = 2
    bin_stim = torch.randn(batch_size, n_listeners, 2, 64,dtype=torch.float64, requires_grad=True)

    estimated = model(bin_stim)

    assert estimated.shape == (batch_size, n_listeners)

def test_llado2025_returns_corresponding_template_angle_with_template_hrirs() -> None:
    # If the model is given a stimulus whose cues exactly match one of the template entries, it should
    # return the corresponding template azimuth.
    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    model = make_model()
    hrirs = get_hrirs_for_template(sofa_path=sofa_path, device=model.device, dtype=model.dtype).unsqueeze(0)
    bin_stim = hrirs.clone().requires_grad_(True)
    n_listeners = hrirs.shape[1]

    estimated = model(bin_stim)

    assert estimated.shape == (1, n_listeners)
    assert torch.allclose(estimated, torch.linspace(-90, 90, 37, device = model.device, dtype = model.dtype).view(1, n_listeners), atol=1e0) 
    # atol=1e0 because the template itd or ild values can be the  same for different angles and the weighting of the frequencies masks this difference 
    # though its only for 1 frequency band usually so the softmax does an average of those
    # Without the frequency weighting, I believe we can reduce the atol to 1e-6


def test_llado2025_returns_corresponding_template_angle_with_scaled_hrirs() -> None:
    # If the model is given a stimulus whose cues exactly match one of the template entries, it should
    # return the corresponding template azimuth.
    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    model = make_model()
    hrirs = get_hrirs_for_template(sofa_path=sofa_path, device=model.device, dtype=model.dtype).unsqueeze(0)
    bin_stim = hrirs.clone().requires_grad_(True) * 5  # Scale the HRIRs slightly, which should not change the estimated angles much.
    n_listeners = hrirs.shape[1]

    estimated = model(bin_stim)

    assert estimated.shape == (1, n_listeners)
    assert torch.allclose(estimated, torch.linspace(-90, 90, 37, device = model.device, dtype = model.dtype).view(1, n_listeners), atol=1e0) 
    # atol=1e0 because the template itd or ild values can be the  same for different angles and the weighting of the frequencies masks this difference 
    # though its only for 1 frequency band usually so the softmax does an average of those
    # Without the frequency weighting, I believe we can reduce the atol to 1e-6
    

def test_llado2025_returns_corresponding_template_angle_with_delayed_hrirs() -> None:
    # If the model is given a stimulus whose cues exactly match one of the template entries, it should
    # return the corresponding template azimuth.
    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    model = make_model()
    hrirs = get_hrirs_for_template(sofa_path=sofa_path, device=model.device, dtype=model.dtype).unsqueeze(0)
    bin_stim = torch.roll(hrirs.clone().requires_grad_(True), shifts=42, dims=-1)  # Delay the HRIRs slightly, which should not change the estimated angles.
    n_listeners = hrirs.shape[1]

    estimated = model(bin_stim)

    assert estimated.shape == (1, n_listeners)
    assert torch.allclose(estimated, torch.linspace(-90, 90, 37, device = model.device, dtype = model.dtype).view(1, n_listeners), atol=1e0) 
    # atol=1e0 because the template itd or ild values can be the  same for different angles and the weighting of the frequencies masks this difference 
    # though its only for 1 frequency band usually so the softmax does an average of those
    # Without the frequency weighting, I believe we can reduce the atol to 1e-6
    

def test_llado2025_returns_corresponding_template_angle_with_truncated_zeros_in_hrirs() -> None:

    # If the model is given a stimulus whose cues exactly match one of the template entries, it should
    # return the corresponding template azimuth.
    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    model = make_model()
    hrirs = get_hrirs_for_template(sofa_path=sofa_path, device=model.device, dtype=model.dtype).unsqueeze(0)
    bin_stim = hrirs.clone().requires_grad_(True)[:,:,:,:-100]  # Truncate the last 100 samples, which are all zeros, which should not change the estimated angles.
    n_listeners = hrirs.shape[1]

    estimated = model(bin_stim)

    assert estimated.shape == (1, n_listeners)
    assert torch.allclose(estimated, torch.linspace(-90, 90, 37, device = model.device, dtype = model.dtype).view(1, n_listeners), atol=1e0) 
    # atol=1e0 because the template itd or ild values can be the  same for different angles and the weighting of the frequencies masks this difference 
    # though its only for 1 frequency band usually so the softmax does an average of those
    # Without the frequency weighting, I believe we can reduce the atol to 1e-6
    

def test_llado2025_returns_corresponding_template_angle_with_scaled_and_delayed_hrirs() -> None:
    # If the model is given a stimulus whose cues exactly match one of the template entries, it should
    # return the corresponding template azimuth.
    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    model = make_model()
    hrirs = get_hrirs_for_template(sofa_path=sofa_path, device=model.device, dtype=model.dtype).unsqueeze(0)
    bin_stim = torch.roll(hrirs * 5, shifts=42, dims=-1)  # Scale and delay the HRIRs slightly, which should not change the estimated angles.
    n_listeners = hrirs.shape[1]    

    estimated = model(bin_stim)

    assert estimated.shape == (1, n_listeners)
    assert torch.allclose(estimated, torch.linspace(-90, 90, 37, device = model.device, dtype = model.dtype).view(1, n_listeners), atol=1e0) # atol=1e0 because the template itd or ild values can be the  same for different angles and the weighting of the frequencies masks this difference though its only for 1 frequency band usually so the softmax does an average of those


def test_llado2025_returns_corresponding_template_angle_with_hrirs_convoluted_with_noise() -> None:
    # If the model is given a stimulus whose cues exactly match one of the template entries, it should
    # return the corresponding template azimuth.
    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    model = make_model()
    hrirs = get_hrirs_for_template(sofa_path=sofa_path, device=model.device, dtype=model.dtype).unsqueeze(0)
    n_lstn = hrirs.shape[1]
    n_samples = hrirs.shape[-1]
    bin_stim = causal_conv1d(hrirs.view(2*n_lstn,1,n_samples), torch.randn(1, 1, 100, device=model.device, dtype=model.dtype)).view(1, n_lstn, 2, n_samples)  # Convolve the HRIRs with noise.

    estimated = model(bin_stim)

    assert estimated.shape == (1, n_lstn)
    assert torch.allclose(estimated, 
                          torch.linspace(-90, 90, 37, device = model.device, dtype = model.dtype).view(1, n_lstn), 
                          atol=5) 
    # atol = 5 degrees, which is a very loose threshold, but the noise should make the estimates quite inaccurate. The point of this test is just to check that the model doesn't break when given noisy stimuli.
    # Without the frequency weighting, I believe we can reduce the atol to 1e-6
    

def test_llado2025_returns_0_azimuth_for_stereo_setup_with_same_stimulus()->None:

    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    fs = 48000
    lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    lstn_idx = torch.tensor([0], dtype=torch.long)
    spk_pos = torch.tensor([[1.732, 1.0], [1.732, -1.0]], dtype=torch.float64)  # Two speakers at the same position, which should produce a stimulus with cues corresponding to 0 degrees azimuth.

    propagation = BinauralFreeFieldPropagation(
        fs=fs,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        sofa_file_path=sofa_path,
        device='cpu',
        dtype=torch.float64
    )

    model = make_model()

    stim = torch.randn(1, 1, 1000, dtype=torch.float64)  # Random stimulus
    spk_filters = torch.ones(1,2,1, dtype=torch.float64)  # Identity filters for both speakers

    bin_stim = propagation(stim, spk_filters, lstn_idx)

    estimated = model(bin_stim)

    assert estimated.shape == (1, 1)
    assert estimated.item() == pytest.approx(0.0, abs=1e-6)


def test_llado2025_returns_loudspeaker_azimuth_for_stereo_setup_with_only_one_loudspeaker_on()->None:

    repo_root = Path(__file__).resolve().parent.parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    fs = 48000
    lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    lstn_idx = torch.tensor([0], dtype=torch.long)
    spk_pos = torch.tensor([[1.732, 1.0], [1.732, -1.0]], dtype=torch.float64)  # Two speakers at the same position, which should produce a stimulus with cues corresponding to 0 degrees azimuth.

    propagation = BinauralFreeFieldPropagation(
        fs=fs,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        sofa_file_path=sofa_path,
        device='cpu',
        dtype=torch.float64
    )

    model = make_model()

    stim = torch.zeros(1, 1, 1000, dtype=torch.float64)  # Random stimulus
    stim[0, 0, 0] = 1.0  # Impulse at the beginning of the stimulus
    spk_filters = torch.zeros(1,2,1, dtype=torch.float64)  # Zero filters for both speakers
    spk_filters[0,0,0] = 1.0  # Identity filter for the first speaker only

    bin_stim = propagation(stim, spk_filters, lstn_idx)

    estimated = model(bin_stim)

    assert torch.allclose(propagation.sources[0].hrirs, bin_stim[0,:,:,:256]), "The HRIRs for the first source should be the same as the first 256 samples of the binaural stimulus, since the stimulus is an impulse and the filter is an identity filter."
    assert estimated.shape == (1, 1)
    assert estimated.item() == pytest.approx(30.0, abs=3), "The estimated azimuth should be approximately 30 degrees."
     # The estimated azimuth should be approximately 30 degrees, which is the azimuth of the first speaker. We use a loose threshold of 3 degrees because the model is not perfect and there may be some small numerical differences in the HRIRs that could affect the estimate. The point of this test is just to check that the model gives a reasonable estimate when only one speaker is on, and that it doesn't break when given a stimulus with very simple cues.