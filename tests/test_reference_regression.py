from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

import numpy as np
import pytest

# Import setup functions
from scripts.setup.setup import set_seed
from scripts.setup.setup import load_gtfb
from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs

# Import sound propagation module
from scripts.propagation.stereo_anechoic import stereo_anechoic

# Import perceptual models
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

# Import neural network related scripts
from scripts.optimisation.nn_init import load_NN
from scripts.optimisation.nn_init import load_optimiser
from scripts.optimisation.regularisation import select_regularistion
# from scripts.optimisation.nn_init import load_scheduler

# Import training script
from scripts.train.train import Train

pytest.importorskip("sofar")
pytest.importorskip("torchaudio")

# Import saving class
from scripts.saving.save_results import saver

def build_perceptual_loss_term(model, device, dtype):
    def loss_term(bin_stim, h_full):  # noqa: ARG001 - kept for a uniform callable interface
        estimation = model(bin_stim)
        target = torch.as_tensor(model.target, device=device, dtype=dtype).expand_as(estimation)
        loss = model.loss(estimation, target) / model.loss_norm
        return loss, estimation

    return loss_term

def build_regularisation_loss_term(regularisation, source_idx):
    def loss_term(bin_stim, h_full):  # noqa: ARG001 - kept for a uniform callable interface
        source_filter = h_full[:, source_idx:source_idx + 1, :]
        loss = regularisation.compute_loss(source_filter) / regularisation.loss_norm
        return loss, None

    return loss_term

def test_reference_100_epoch_matches_known_output() -> None:

    """
    Runs the exact same code as: python main.py --epochs=100 --n_samples=10 --target_angle=15
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
        near_field=0,
        epochs=100,
        n_samples=10,
        target_angle=15.0,
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
    stereo_prop = stereo_anechoic(stim=stim, lstn_pos=lstn_pos, spk_pos=spk_pos, fs=fs, sofa_file_path=args.sofa_path) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_loss = F.mse_loss
    unc_loss = F.l1_loss
    col_loss = F.l1_loss

    loc_target = args.target_angle
    unc_target = 0
    col_target = 0

    
    loc_model = llado2025(loss = loc_loss, target = loc_target, hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = col_loss, target = col_target, stim=stim, hrirs=hrirs, target_angle=args.target_angle, fs=fs).to(device=device, dtype=dtype) # Spectral Matching model
    unc_model = desena2020(loss = unc_loss, target = unc_target, hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # De Sena et al. 2020 model
    

    perceptual_dict = {'localisation': loc_model,
                         'uncertainty': unc_model,
                         'colouration': col_model}
    loss_terms = {
        key: build_perceptual_loss_term(model, device=device, dtype=dtype)
        for key, model in perceptual_dict.items()
    }
    regularisation = select_regularistion(args.reg, n_samples=args.n_samples, device=device, dtype=dtype)
    if regularisation is not None:
        for source_idx in range(len(spk_pos)):
            loss_terms[f"regularisation_source_{source_idx}"] = build_regularisation_loss_term(
                regularisation,
                source_idx=source_idx,
            )

    # Initialise neural network
    nn_model = load_NN(n_samples=args.n_samples, device=device, dtype=dtype, init = args.init) # Load neural network ICLD or FIR
    nn_optimiser = load_optimiser(nn_model, lr=args.lr) # Load optimiser for neural network
    # nn_scheduler = load_scheduler(nn_optimiser, epochs)

## Initliase saving folder and logger

    saver_obj = saver(perceptual_dict, lstn_pos, args)
    logger = saver_obj if args.log else None

## Training loop

    # Train the neural network
    trainer = Train(n_epoch=epochs,
                    nn=nn_model,
                    prop=stereo_prop,
                    perceptual_dict=perceptual_dict,
                    loss_terms=loss_terms,
                    optimiser=nn_optimiser,
                    loggers = [logger, None], 
                    scheduler=None) # Initialize training class
    trainer.train()
    
## Print and Plot results

    full_perceptual_dict = {'localisation': loc_model,
                         'uncertainty': unc_model,
                         'colouration': col_model}

    # Evaluate final neural network FIRs
    est_filters, estimation_dict, bin_stim = trainer.evaluate(full_perceptual_dict)
    # Normalise filters to prevent clipping
    est_filters /= np.max(np.abs(est_filters)) / 0.9  # TODO: move this to a more appropriate place at the end of the training loop

    # Expected from: python main.py --epochs=100 --n_samples=10 --target_angle=15 (with device=cpu)
    expected_path = Path(__file__).resolve().parent / "reference_outputs" / "test_reference_100_epoch_matches_known_output.json"
    with open(expected_path) as f:
        expected = json.load(f)

    expected_filters = np.array(expected["est_filters"], dtype=np.float32)
    expected_localisation = expected["localisation"]
    expected_colouration = expected["colouration"]
    expected_uncertainty = expected["uncertainty"]

    filters_np = np.asarray(est_filters, dtype=np.float32)
    np.testing.assert_allclose(filters_np, expected_filters, rtol=1e-2)
    np.testing.assert_allclose(np.asarray(estimation_dict["localisation"]), expected_localisation, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(estimation_dict["colouration"]), expected_colouration, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(estimation_dict["uncertainty"]), expected_uncertainty, rtol=1e-4, atol=1e-4)
