# Enable MPS fallback for PyTorch on macOS
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import interface libraries
from pathlib import Path
from types import SimpleNamespace

# Import computational libraries
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from scripts.optimisation.custom_modules import MultiSourceFIR

# Import libraries for plotting
import matplotlib.pyplot as plt
from scripts.plotting.graph_plot import plot_time
from scripts.plotting.graph_plot import plot_freq
from scripts.plotting.dense_arrows_plot import plot_arrows
from scripts.plotting.graph_plot import plot_color

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
from scripts.train.train import Train

# Import saving class
from scripts.saving.save_results import saver

# Import time
import time

def build_perceptual_loss_term(model, device, dtype):
    def loss_term(bin_stim, h_full):  # noqa: ARG001 - kept for a uniform callable interface
        model_estimation = model(bin_stim)
        target = torch.as_tensor(model.target, device=device, dtype=dtype).expand_as(model_estimation)
        loss = model.loss(model_estimation, target) / model.loss_norm
        return loss, model_estimation

    return loss_term

def build_regularisation_loss_term(regularisation, source_idx):
    def loss_term(bin_stim, h_full):  # noqa: ARG001 - kept for a uniform callable interface
        source_filter = h_full[:, source_idx:source_idx + 1, :]
        loss = regularisation.compute_loss(source_filter) / regularisation.loss_norm
        return loss, None

    return loss_term


if __name__ == "__main__":
    near_field=0
    epochs=100
    n_samples=10
    target_angle=15.0
    selected_stim="random"
    repo_root = Path(__file__).resolve().parent
    sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa")
    sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv")
    lr=0.001
    seed=46
    log=1
    device="cpu"
    init=1
    fs = 48000

## Initial setup
    # Set data type, device and sampling frequency
    dtype = torch.float32
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    if seed is not None:
        set_seed(seed=seed)

    # Loudspkeaker and listener positions
    lstn_pos = [[0.0, 0], [0.0, 0.5], [0.0, -0.5]] # 3 listener positions
    spk_pos = [[1.732,1], [1.732,-1]]

## Load data
    gtfb = load_gtfb(fs=fs, device=device, dtype=dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=fs, device=device, dtype=dtype, selected_stim=selected_stim)  # Load stimulus
    hrirs = load_hrirs(dtype=dtype, device=device, sofa_path=sofa_path, idx_path=sofa_idx_path) # Load HRIRs using specific location from SOFA file

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(stim=stim, lstn_pos=lstn_pos, spk_pos=spk_pos, fs=fs, sofa_file_path=sofa_path) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_loss = F.mse_loss
    unc_loss = F.l1_loss
    col_loss = F.l1_loss

    loc_target = target_angle
    unc_target = 0
    col_target = 0

    near_field_kwargs = {"lstn_pos_conf": lstn_pos} if near_field == 1 else {}
    loc_model = llado2025(loss = loc_loss, target = loc_target, hrirs=hrirs, fs=fs, gtfb=gtfb, **near_field_kwargs).to(device=device, dtype=dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = col_loss, target = col_target, stim=stim, hrirs=hrirs, target_angle=target_angle, fs=fs, **near_field_kwargs).to(device=device, dtype=dtype) # Spectral Matching model
    unc_model = desena2020(loss = unc_loss, target = unc_target, hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # De Sena et al. 2020 model

    perceptual_dict = {'localisation': loc_model,
                         'uncertainty': unc_model,
                         'colouration': col_model}
    loss_terms = {
        key: build_perceptual_loss_term(model, device=device, dtype=dtype)
        for key, model in perceptual_dict.items()
    }
    
    regularisation = second_moment_regularisation(n_samples=n_samples, device=device, dtype=dtype)
    if regularisation is not None:
        for source_idx in range(len(spk_pos)):
            loss_terms[f"regularisation_source_{source_idx}"] = build_regularisation_loss_term(
                regularisation,
                source_idx=source_idx,
            )

    # Initialise neural network

    nn_model = MultiSourceFIR(
        n_samples=n_samples,
        n_sources=2,
        layer_cls=nn.Linear,
        layer_kwargs={"bias": False},
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[0, 0] = 1.0

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=lr)
    
## Initliase saving folder and logger

    saver_obj = saver(perceptual_dict, lstn_pos)
    logger = saver_obj if log else None

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
    time_train_start = time.time()
    trainer.train()
    time_train = time.time() - time_train_start
    print(f"Time for full optimisation:{time_train:.2f}")

## Print and Plot results
    full_perceptual_dict = {'localisation': loc_model,
                         'uncertainty': unc_model,
                         'colouration': col_model}

    # Evaluate final neural network FIRs
    est_filters, estimation_dict, bin_stim = trainer.evaluate(full_perceptual_dict)
    # Normalise filters to prevent clipping 
    est_filters /= np.max(np.abs(est_filters)) / 0.9 # TODO: move this to a more appropriate place at the end of the training loop

    # Print estimated perceptual metrics
    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    # Plot estimated filters (time and frequency domain)
    time_plot = plot_time(est_filters, show=False)
    freq_plot = plot_freq(est_filters, fs, show=False)
    arrow_plot = None
    if 'localisation' in full_perceptual_dict:
        if full_perceptual_dict['localisation'].lstn_pos_conf is not None:
            arrow_plot = plot_arrows(stereo_prop.lstn_pos, 
                                        stereo_prop.spk_pos, 
                                        torch.tensor(estimation_dict['localisation'], device=device, dtype=dtype), 
                                        torch.tensor(estimation_dict['uncertainty'], device=device, dtype=dtype) if 'uncertainty' in estimation_dict else None, 
                                        torch.tensor(estimation_dict['colouration'],device=device, dtype=dtype) if 'colouration' in estimation_dict else None, 
                                        loc_target=torch.tensor(target_angle),
                                        show=False
            )
        else:
            arrow_plot = plot_arrows(stereo_prop.lstn_pos, 
                                        stereo_prop.spk_pos, 
                                        torch.tensor(estimation_dict['localisation'], device=device,dtype=dtype), 
                                        torch.tensor(estimation_dict['uncertainty'],device=device,dtype=dtype) if 'uncertainty' in estimation_dict else None, 
                                        torch.tensor(estimation_dict['colouration'],device=device,dtype=dtype) if 'colouration' in estimation_dict else None, 
                                        show=False
            )
    color_l_plot, color_r_plot = plot_color(bin_stim, col_model.reference_stim, stereo_prop.lstn_pos)
    plt.show()

## Save results

    saver_obj.save_results(estimation_dict=estimation_dict, est_filters=est_filters, fs=fs, lstn_pos=lstn_pos, spk_pos=spk_pos, trainer=trainer)
    saver_obj.save_plots(time_plot,freq_plot,arrow_plot, color_l_plot, color_r_plot)

    # Save per-epoch source FIR evolution as videos (one per source)
    # saver_obj.save_filter_video(trainer.h_history, source_index=0, fs=fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=1, fs=fs)
