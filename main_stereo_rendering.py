# Enable MPS fallback for PyTorch on macOS
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import interface libraries
from pathlib import Path
import time
from types import SimpleNamespace

# Import computational libraries
import numpy as np
import torch
import torch.nn.functional as F

# Import project modules
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.plotting.results_plot import plot_results
from scripts.setup.setup import load_gtfb, load_hrirs, load_stim, set_seed
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
from scripts.saving.save_results import saver


def run_experiment(loss_dict_builder, run_label):
    repo_root = Path(__file__).resolve().parent
    d =100
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=1500,
        num_lstn_pos=3,
        listener_batch_size=None,  # 5 mirrored pairs plus the centre listener
        symmetric_listener_sampling=True,
        n_samples=128,
        target_pos=[d, float(np.tan(np.deg2rad(0.0)) * d)],
        selected_stim="random",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.001,
        seed=46,
        log=1,
        fs=48000,
    )
    env.dtype = torch.float64
    env.device_str = "cpu"
    env.device = torch.device(env.device_str)

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    # n_per_segment = env.num_lstn_pos // 4
    # segments = [
    #     np.linspace(-0.5, -0.325, n_per_segment),
    #     np.linspace(-0.225, -0.05, n_per_segment),
    #     np.linspace(0.05, 0.225, n_per_segment),
    #     np.linspace(0.325, 0.5, n_per_segment),
    # ]
    # y_positions = np.concatenate(segments)
    # env.lstn_pos = [[0.0, float(y)] for y in y_positions]
    env.lstn_pos = [[0.0, float(y)] for y in np.linspace(-0.5, 0.5, env.num_lstn_pos)] 
    # env.lstn_pos = [[0.0, 0.0]] # Only the centre position for now, as the spectral matching model only supports one listener position.
    base_angle_rad = np.deg2rad(30.0)
    env.speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
    env.spk_pos = [
        [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
        [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
    ]
    env.n_sources = len(env.spk_pos)

    # Reset the seed so each run starts from the same point.
    set_seed(seed=env.seed)

## Load data
    gtfb = load_gtfb(fs=env.fs, device=env.device, dtype=env.dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=env.fs, device=env.device, dtype=env.dtype, selected_stim=env.selected_stim)  # Load stimulus
    hrirs = load_hrirs(dtype=env.dtype, device=env.device, sofa_path=env.sofa_path, idx_path=env.sofa_idx_path) # Load HRIRs using specific location from SOFA file

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(stim=stim, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, fs=env.fs, sofa_file_path=env.sofa_path) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos).to(device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, stim=stim, hrirs=hrirs, target_pos=env.target_pos, fs=env.fs).to(device=env.device, dtype=env.dtype) # Spectral Matching model
    unc_model = desena2020(loss = F.l1_loss, target = 0, fs=env.fs, sofa_path=env.sofa_path).to(device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
    regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)
    loss_dict = loss_dict_builder(loc_model, col_model, unc_model, regularisation, env.device, env.dtype)

    # Initialise neural network
    nn_model = MultiSourceFIR(
        n_samples=env.n_samples,
        n_sources=env.n_sources,
    ).to(device=env.device, dtype=env.dtype)
    # Initialise FIR filter weights
    with torch.no_grad():
        for i, branch in enumerate(nn_model.branches):
            branch.weight.fill_(0.0)
            branch.weight[0, 0] = 1.0

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

## Initliase saving folder and logger
    saver_obj = saver(loss_dict, env.lstn_pos)
    logger = saver_obj if env.log else None

## Training loop
    # Train the neural network
    trainer = Train(n_epoch=env.epochs,
                    nn=nn_model,
                    prop=stereo_prop,
                    loss_dict=loss_dict,
                    optimiser=nn_optimiser,
                    loggers = [logger, None], 
                    listener_batch_size=env.listener_batch_size,
                    symmetric_listener_sampling=env.symmetric_listener_sampling) # Initialize training class
    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

## Print and Plot results
    # Evaluate final neural network FIRs
    eval_perceptual_dict = {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1.0, device=env.device, dtype=env.dtype),
            "uncertainty": ListenerBasedLoss(model=unc_model, weight=1, device=env.device, dtype=env.dtype),
            "colouration": ListenerBasedLoss(model=col_model, weight=1, device=env.device, dtype=env.dtype)
        }
    eval_lstn_pos = [[0.0 , 0.0], [0.0, 0.5], [0.0, -0.5]]
    stereo_prop_eval = BinauralFreeFieldPropagation(stim=stim, lstn_pos=eval_lstn_pos, spk_pos=env.spk_pos, fs=env.fs, sofa_file_path=env.sofa_path, apply_distance_attenuation=True)
    trainer.prop = stereo_prop_eval
    est_filters, estimation_dict, bin_stim = trainer.evaluate(loss_dict=eval_perceptual_dict)
    # Normalise filters to prevent clipping 
    est_filters /= np.max(np.abs(est_filters)) / 0.9 # TODO: move this to a more appropriate place at the end of the training loop

    # Print estimated perceptual metrics
    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    saver_obj.save_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        fs=env.fs,
        lstn_pos=eval_lstn_pos,
        spk_pos=env.spk_pos,
        trainer=trainer,
        env=env,
    )
    plots = plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=bin_stim,
        prop_model=stereo_prop_eval,
        loss_dict=eval_perceptual_dict,
        target_pos=env.target_pos,
        stim=stim,
        fs=env.fs,
        device=env.device,
        dtype=env.dtype,
        show=True,
    )
    saver_obj.save_plots(plots)

    # Save per-epoch source FIR evolution as videos (one per source)
    # saver_obj.save_filter_video(trainer.h_history, source_index=0, fs=env.fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=1, fs=env.fs)


if __name__ == "__main__":
    run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "uncertainty": ListenerBasedLoss(model=unc_model, weight=1.0e0, device=device, dtype=dtype),
        },
        run_label="uncertaintyS_only",
    )
