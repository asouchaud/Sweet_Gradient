from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from scripts.optimisation.custom_modules import MultiSourceFIR

import numpy as np
import pytest

# Import setup functions
from scripts.setup.setup import set_seed
from scripts.setup.setup import load_gtfb
from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs

# Import sound propagation module
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation

# Import perceptual models
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

# Import neural network related scripts
from scripts.optimisation.regularisation import first_moment_regularisation
from scripts.optimisation.regularisation import second_moment_regularisation

# Import training script
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train

pytest.importorskip("sofar")
pytest.importorskip("torchaudio")

# Import saving class
from scripts.saving.save_results import saver

def select_regularistion(reg: str, n_samples: int, device='cuda', dtype=torch.float32):
    if reg == 0:
        return None
    if reg == 1:
        return first_moment_regularisation(n_samples=n_samples, device=device, dtype=dtype)
    if reg == 2:
        return second_moment_regularisation(n_samples=n_samples, device=device, dtype=dtype)
    return None

def test_reference_100_epoch_matches_known_output() -> None:

    """
    Runs the same legacy 15 degree reference case, now expressed via ``target_pos``,
    with device=cpu for determinism. We chdir to repo root and call main() so there is no
    duplicated logic or path/cwd mismatch.
    """
    """Run the pipeline. If return_results is True, return (est_filters, estimation_dict) and skip plot/save."""
## Initial setup

    # Default paths relative to this repository (portable across machines/OSes)
    repo_root = Path(__file__).resolve().parents[1]
    default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
    default_idx_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"
    args = SimpleNamespace(
        epochs=100,
        n_samples=10,
        target_pos=[1.732, float(np.tan(np.deg2rad(15.0)) * 1.732)],
        stimulus="random",
        sofa_path=default_sofa_path,
        idx_path=default_idx_path,
        lstn_pos_conf=1,
        lr=0.001,
        init=1,
        reg=2,
        alpha=1.5,
        seed=46,
        log=1,
        device="cpu",
    )

    # Set data type, device and sampling frequency
    dtype = torch.float32
    device = torch.device(args.device)
    fs = 48000
    set_seed(seed=args.seed)

    # Training parameters
    epochs = args.epochs # Number of training epochs

    # Loudspkeaker and listener positions
    lstn_pos = [[0.0, 0], [0.0, 0.5], [0.0, -0.5]]  # 3 listener positions
    spk_pos = [[1.732, 1], [1.732, -1]]

## Load data

    gtfb = load_gtfb(fs=fs, device=device, dtype=dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=fs, device=device, dtype=dtype, selected_stim=args.stimulus)  # Load stimulus
    hrirs = load_hrirs(dtype=dtype, device=device, sofa_path=args.sofa_path, idx_path=args.idx_path) # Load HRIRs using specific location from SOFA file

## Initialize model

    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(stim=stim, lstn_pos=lstn_pos, spk_pos=spk_pos, fs=fs, sofa_file_path=args.sofa_path) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_loss = F.mse_loss
    unc_loss = F.l1_loss
    col_loss = F.l1_loss

    unc_target = 0
    col_target = 0

    
    loc_model = llado2025(loss = loc_loss, target_pos=args.target_pos, hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = col_loss, target = col_target, stim=stim, hrirs=hrirs, target_pos=args.target_pos, fs=fs).to(device=device, dtype=dtype) # Spectral Matching model
    unc_model = desena2020(loss = unc_loss, target = unc_target, hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # De Sena et al. 2020 model
    

    loss_dict = {
        "localisation": ListenerBasedLoss(model=loc_model, weight=1.0, device=device, dtype=dtype),
        "uncertainty": ListenerBasedLoss(model=unc_model, weight=1.0e1, device=device, dtype=dtype),
        "colouration": ListenerBasedLoss(model=col_model, weight=1.0e-2, device=device, dtype=dtype),
    }
    regularisation = select_regularistion(args.reg, n_samples=args.n_samples, device=device, dtype=dtype)
    if regularisation is not None:
        for source_idx in range(len(spk_pos)):
            reg_key = f"regularisation_source_{source_idx}"
            loss_dict[reg_key] = SourceBasedLoss(
                regulariser=regularisation,
                source_idx=source_idx,
                weight=1.0e-3,
            )

    # Initialise neural network
    nn_model = MultiSourceFIR(
        n_samples=args.n_samples,
        n_sources=2,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[0, 0] = 1.0

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    
## Initliase saving folder and logger

    saver_obj = saver(loss_dict, lstn_pos)
    logger = saver_obj if args.log else None

## Training loop

    # Train the neural network
    trainer = Train(n_epoch=epochs,
                    nn=nn_model,
                    prop=stereo_prop,
                    loss_dict=loss_dict,
                    optimiser=nn_optimiser,
                    loggers = [logger, None], 
                    scheduler=None) # Initialize training class
    trainer.train()
    
## Print and Plot results

    # Evaluate final neural network FIRs
    est_filters, estimation_dict, bin_stim = trainer.evaluate()
    # Normalise filters to prevent clipping
    est_filters /= np.max(np.abs(est_filters)) / 0.9  # TODO: move this to a more appropriate place at the end of the training loop

    saver_obj.save_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        fs=fs,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        trainer=trainer,
    )

    # Expected from the legacy 15 degree reference case, now expressed as target_pos.
    expected_path = Path(__file__).resolve().parent / "reference_outputs" / "test_reference_100_epoch_matches_known_output.json"
    with open(expected_path) as f:
        expected = json.load(f)

    expected_filters = np.array(expected["est_filters"], dtype=np.float32)
    expected_localisation = expected["localisation"]
    expected_colouration = expected["colouration"]
    expected_uncertainty = expected["uncertainty"]

    filters_np = np.asarray(est_filters, dtype=np.float32)
    _ci = os.environ.get("GITHUB_ACTIONS") == "true"
    np.testing.assert_allclose(filters_np, expected_filters, rtol=1e-2 if _ci else 1e-5, atol=1e-2 if _ci else 1e-5)
    np.testing.assert_allclose(np.asarray(estimation_dict["localisation"]), expected_localisation, rtol=1e-2 if _ci else 1e-4, atol=1e-0 if _ci else 1e-2)
    np.testing.assert_allclose(np.asarray(estimation_dict["colouration"]), expected_colouration, rtol=1e-2 if _ci else 1e-4, atol=1e-0 if _ci else 1e-2)
    np.testing.assert_allclose(np.asarray(estimation_dict["uncertainty"]), expected_uncertainty, rtol=1e-2 if _ci else 1e-5, atol=1e-2 if _ci else 1e-5)