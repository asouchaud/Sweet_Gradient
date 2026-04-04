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
import matplotlib.pyplot as plt

# Import project modules
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.plotting.results_plot import plot_pair_directivity_polar, plot_results, plot_arrows_with_directivity
from scripts.setup.setup import load_gtfb, load_hrirs, load_stim, set_seed
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
from scripts.saving.save_results import saver


def build_listener_position_weights(lstn_pos, y_pos_lo, y_pos_hi, y_neg_lo, y_neg_hi, w_high, w_low):
    """
    Weight each listener by its global y coordinate only: high in [y_pos_lo, y_pos_hi] and
    [y_neg_lo, y_neg_hi], low elsewhere (e.g. centre strip and far lateral positions).
    """
    listener_weights = []
    for listener_pos in lstn_pos:
        y = float(listener_pos[1])
        in_pos_band = y_pos_lo <= y <= y_pos_hi
        in_neg_band = y_neg_lo <= y <= y_neg_hi
        listener_weights.append(w_high if (in_pos_band or in_neg_band) else w_low)
    return np.asarray(listener_weights, dtype=np.float32)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=200,
        n_samples=41,
        num_lstn_pos=51,
        listener_batch_size=None, 
        symmetric_listener_sampling=True,
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.05,
        seed=46,
        log=1,
        fs=48000,
        directivity_radius_m=10.0,
        directivity_freqs_hz=[250, 500, 1000, 2000, 4000],
        directivity_db_floor=-40,
        speakers_per_array=2,
        speaker_distance=2.5,  # Distance from each array centre to the reference-system center (0, 0)
        array_centre_angles_deg=[30.0, -30.0],
        speaker_spacing_m=0.10,  # Adjacent spacing within each array (global y-axis)
        apply_distance_attenuation=True,
        listener_weight_y_pos_lo=0.20,
        listener_weight_y_pos_hi=1.25,
        listener_weight_y_neg_lo=-1.25,
        listener_weight_y_neg_hi=-0.20,
        listener_weight_w_high=1.0,
        listener_weight_w_low=0.0,
    )
    env.dtype = torch.float32
    env.device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env.device = torch.device(env.device_str)

    env.target_pos=[float(env.speaker_distance * np.cos(np.deg2rad(env.array_centre_angles_deg[0]))), 0.0]
    # env.target_pos=[10.5, 0.0]
    print(env.target_pos)

## Initial setup
    print(f"Using device: {env.device}")

    # Set random seed for reproducibility
    set_seed(seed=env.seed)

    # Loudspkeaker and listener positions
    env.lstn_pos = [[0.0, float(y)] for y in np.linspace(-1.25, 1.25, env.num_lstn_pos)]  # 10 listener positions

    relative_offsets = (np.arange(env.speakers_per_array, dtype=float) - (env.speakers_per_array - 1) / 2.0) * env.speaker_spacing_m
    upper_angle_deg, lower_angle_deg = env.array_centre_angles_deg
    upper_centre = np.array(
        [
            env.speaker_distance * np.cos(np.deg2rad(upper_angle_deg)),
            env.speaker_distance * np.sin(np.deg2rad(upper_angle_deg)),
        ],
        dtype=float,
    )
    lower_centre = np.array(
        [
            env.speaker_distance * np.cos(np.deg2rad(lower_angle_deg)),
            env.speaker_distance * np.sin(np.deg2rad(lower_angle_deg)),
        ],
        dtype=float,
    )

    # Build two arrays mirrored around the x-axis:
    # upper uses +offset, lower uses -offset for each corresponding index.
    upper_array = [[float(upper_centre[0]), float(upper_centre[1] + off)] for off in relative_offsets]
    lower_array = [[float(lower_centre[0]), float(lower_centre[1] - off)] for off in relative_offsets]
    env.spk_pos = upper_array + lower_array
    env.n_sources = len(env.spk_pos)

    listener_weights = build_listener_position_weights(
        lstn_pos=env.lstn_pos,
        y_pos_lo=env.listener_weight_y_pos_lo,
        y_pos_hi=env.listener_weight_y_pos_hi,
        y_neg_lo=env.listener_weight_y_neg_lo,
        y_neg_hi=env.listener_weight_y_neg_hi,
        w_high=env.listener_weight_w_high,
        w_low=env.listener_weight_w_low,
    )
    if float(np.sum(listener_weights)) <= 0.0:
        listener_weights = np.ones(len(env.lstn_pos), dtype=np.float32)
    env.listener_weights = listener_weights.tolist()

## Load data
    gtfb = load_gtfb(fs=env.fs, device=env.device, dtype=env.dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=env.fs, device=env.device, dtype=env.dtype, selected_stim=env.selected_stim)  # Load stimulus
    hrirs = load_hrirs(dtype=env.dtype, device=env.device, sofa_path=env.sofa_path, idx_path=env.sofa_idx_path) # Load HRIRs using specific location from SOFA file

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(
        stim=stim, 
        lstn_pos=env.lstn_pos, 
        spk_pos=env.spk_pos, 
        fs=env.fs, 
        sofa_file_path=env.sofa_path, 
        apply_distance_attenuation=env.apply_distance_attenuation) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, hrirs=hrirs, fs=env.fs, gtfb=gtfb, lstn_pos_conf=env.lstn_pos).to(device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, stim=stim, hrirs=hrirs, target_pos=env.target_pos, fs=env.fs, lstn_pos_conf=env.lstn_pos).to(device=env.device, dtype=env.dtype) # Spectral Matching model
    unc_model = desena2020(loss = F.l1_loss, target = 0, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
    regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)

    env.loc_weight = 1.0
    env.unc_weight = 1.0e1
    env.col_weight = 1e-01
    env.reg_weight = 1.0e-2

    listener_loss_kwargs = dict(device=env.device, dtype=env.dtype, listener_weights=listener_weights)

    loss_dict = {
        "localisation": ListenerBasedLoss(model=loc_model, weight=env.loc_weight, **listener_loss_kwargs),
        # "uncertainty": ListenerBasedLoss(model=unc_model, weight=env.unc_weight, **listener_loss_kwargs),
        # "colouration": ListenerBasedLoss(model=col_model, weight=env.col_weight, **listener_loss_kwargs),
        # "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=env.reg_weight),
        # "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=env.reg_weight),
        # "regularisation_source_2": SourceBasedLoss(regulariser=regularisation, source_idx=2, weight=env.reg_weight),
        # "regularisation_source_3": SourceBasedLoss(regulariser=regularisation, source_idx=3, weight=env.reg_weight),
    }

    # Initialise neural network
    nn_model = MultiSourceFIR(
        n_samples=env.n_samples,
        n_sources=env.n_sources,
    ).to(device=env.device, dtype=env.dtype)
    # Initialise FIR filter weights
    centre_tap_idx = env.n_samples // 2
    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[centre_tap_idx, 0] = 1.0

    # Tie mirrored loudspeakers between the two arrays to preserve symmetry.
    for local_idx in range(env.speakers_per_array):
        nn_model.branches[env.speakers_per_array + local_idx] = nn_model.branches[local_idx]
    
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
                    scheduler=None,
                    listener_batch_size=env.listener_batch_size,
                    symmetric_listener_sampling=env.symmetric_listener_sampling) # Initialize training class
    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

## Print and Plot results
    # Evaluate final neural network FIRs
    est_filters, estimation_dict, bin_stim = trainer.evaluate()
    # Normalise filters to prevent clipping 
    est_filters /= np.max(np.abs(est_filters)) / 0.9 # TODO: move this to a more appropriate place at the end of the training loop
    est_filters = est_filters * 1.41

    # Print estimated perceptual metrics
    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    saver_obj.save_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        fs=env.fs,
        lstn_pos=env.lstn_pos,
        spk_pos=env.spk_pos,
        trainer=trainer,
        env=env,
    )
    plots = plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=bin_stim,
        prop_model=stereo_prop,
        loss_dict=loss_dict,
        target_pos=env.target_pos,
        stim=stim,
        fs=env.fs,
        device=env.device,
        dtype=env.dtype,
        show=False,
    )

    fig_listener_weights, ax_listener_weights = plt.subplots(figsize=(6, 3))
    lstn_y = np.asarray([pos[1] for pos in env.lstn_pos], dtype=float)
    ax_listener_weights.plot(lstn_y, listener_weights, linewidth=2)
    ax_listener_weights.set_xlabel("Listener position y (m)")
    ax_listener_weights.set_ylabel("Listener weight")
    ax_listener_weights.set_title("Listener-position weights")
    ax_listener_weights.grid(True, alpha=0.25)
    plots["listener_weights"] = fig_listener_weights

    rodenas2003_fig12_opt = np.loadtxt(repo_root / "data" / "miscellaneous" / "rodenas2003_fig12_opt.csv", delimiter=",", skiprows=1)
    rodenas2003_fig12_opt = [[-((angle_deg + 180) % 360.0), level_db] for angle_deg, level_db in rodenas2003_fig12_opt]
    rodenas2003_fig12_1000Hz = np.loadtxt(repo_root / "data" / "miscellaneous" / "rodenas2003_fig12_1000Hz.csv", delimiter=",", skiprows=1)
    rodenas2003_fig12_1000Hz = [[-((angle_deg + 180) % 360.0), level_db] for angle_deg, level_db in rodenas2003_fig12_1000Hz]
    
    right_array_indices = tuple(range(env.speakers_per_array, 2 * env.speakers_per_array))
    left_array_indices = tuple(range(env.speakers_per_array))
    plots["directivity_plot"] = plot_pair_directivity_polar(
        est_filters=est_filters,
        spk_pos=env.spk_pos,
        fs=env.fs,
        group_indices=right_array_indices,
        radius=env.directivity_radius_m,
        freqs_eval_hz=env.directivity_freqs_hz,
        overlay_curves=[rodenas2003_fig12_opt, rodenas2003_fig12_1000Hz],
        overlay_labels=["rodenas2003 (opt.)", "rodenas2003 (1 kHz)"],
        db_floor=env.directivity_db_floor,
        show=False,
    )

    plots["arrows_and_directivity"] = plot_arrows_with_directivity(
        est_filters = est_filters,
        spk_pos = torch.tensor(env.spk_pos),
        lstn_pos = torch.tensor(env.lstn_pos),
        loc = torch.tensor(estimation_dict["localisation"]),
        fs = env.fs,
        group_idx = [left_array_indices, right_array_indices],
        unc=None,
        col=None,
        target_pos=env.target_pos,
        show_listener_target_lines=True,
        vbap_values=None,
        show=False,
        n_angles=360,
        freqs_eval_hz=[250, 500, 1000, 2000, 4000],
        directivity_observation_radius_m=env.directivity_radius_m,
        c=343.0,
        directivity_mode="avg",   # "avg" or "single"
        directivity_freq_idx=2,   # used if mode="single"
        directivity_scale=0.8,
        directivity_floor_db=-20,
        overlay_curves=[rodenas2003_fig12_opt, rodenas2003_fig12_1000Hz],
        overlay_labels=["rodenas2003 (optimum)", "rodenas2003 (1000 Hz)"],
        legend_bbox_to_anchor=(0.28, 1.02),
        figsize=(5, 5),
    )

    npa = env.speakers_per_array
    est_filters_centre = np.stack(
        [
            np.sum(est_filters[0:npa], axis=0),
            np.sum(est_filters[npa : 2 * npa], axis=0),
        ],
        axis=0,
    )
    spk_pos_centre = np.stack([upper_centre, lower_centre], axis=0)
    stereo_prop_centre = BinauralFreeFieldPropagation(
        stim=stim,
        lstn_pos=env.lstn_pos,
        spk_pos=[upper_centre.tolist(), lower_centre.tolist()],
        fs=env.fs,
        sofa_file_path=env.sofa_path,
        apply_distance_attenuation=env.apply_distance_attenuation,
    ).to(device=env.device, dtype=env.dtype)
    loc_model.eval()
    with torch.no_grad():
        h_centre = torch.as_tensor(est_filters_centre, device=env.device, dtype=env.dtype).unsqueeze(0)
        bin_stim_centre = stereo_prop_centre(h_centre)
        loc_centre = loc_model(bin_stim_centre).squeeze(0)

    plots["arrows_and_directivity_centre"] = plot_arrows_with_directivity(
        est_filters=est_filters_centre,
        spk_pos=torch.as_tensor(spk_pos_centre, device=env.device, dtype=env.dtype),
        lstn_pos=torch.tensor(env.lstn_pos, device=env.device, dtype=env.dtype),
        loc=loc_centre,
        fs=env.fs,
        group_idx=[(0,), (1,)],
        unc=None,
        col=None,
        target_pos=None,
        show_listener_target_lines=False,
        vbap_values=None,
        show=False,
        n_angles=360,
        freqs_eval_hz=list(env.directivity_freqs_hz),
        directivity_observation_radius_m=env.directivity_radius_m,
        c=343.0,
        directivity_mode="avg",
        directivity_freq_idx=2,
        directivity_scale=0.8,
        directivity_floor_db=-20,
        directivity_label="Omni source",
        legend_ncol=1,
        figsize=(5, 5),
    )

    saver_obj.save_plots(plots)

    # Save per-epoch source FIR evolution as videos (one per source)
    # saver_obj.save_filter_video(trainer.h_history, source_index=0, fs=env.fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=1, fs=env.fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=2, fs=env.fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=3, fs=env.fs)
    # saver_obj.save_directivity_video(
    #     trainer.h_history,
    #     spk_pos=env.spk_pos,
    #     fs=env.fs,
    #     group_indices=right_array_indices,
    #     radius=env.directivity_radius_m,
    #     freqs_eval_hz=env.directivity_freqs_hz,
    #     overlay_curves=[rodenas2003_fig12_opt, rodenas2003_fig12_1000Hz],
    #     overlay_labels=["rodenas2003 (optimum)", "rodenas2003 (1000 Hz)"],
    #     db_floor=env.directivity_db_floor,
    # )
