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
from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.datasets.listener_dataset import get_listener_loader
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.plotting.results_plot import plot_pair_directivity_polar, plot_results, plot_arrows_with_directivity
from scripts.setup.setup import set_seed, build_eval_objects
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.perceptual_models.ipd.ipd import interaural_phase_difference_matching
from scripts.optimisation.regularisation import second_moment_regularisation, source_flatness
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
        listener_batch_size=51, 
        symmetric_listener_sampling=True,
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
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
        speaker_spacing_m=0.145,  # Adjacent spacing within each array (global y-axis)
        apply_distance_attenuation=True,
        listener_weight_y_pos_lo=0.20,
        listener_weight_y_pos_hi=1.25,
        listener_weight_y_neg_lo=-1.25,
        listener_weight_y_neg_hi=-0.20,
        listener_weight_w_high=1.0,
        listener_weight_w_low=0.0,
    )
    env.dtype = torch.float32
    env.device_str = "cpu"
    env.device = torch.device(env.device_str)
    env.stimulus_batch_size = 1
    env.n_filter_taps = env.n_samples

    d = 2
    env.target_pos=torch.tensor([d, float(np.tan(np.deg2rad(0.0)) * d)], device=env.device, dtype=env.dtype)
    # env.target_pos=[10.5, 0.0]
    print(env.target_pos)

## Initial setup
    print(f"Using device: {env.device}")

    # Set random seed for reproducibility
    set_seed(seed=env.seed)

    # Loudspkeaker and listener positions
    env.lstn_pos = torch.tensor([[0.0, float(y)] for y in np.linspace(-1.25, 1.25, env.num_lstn_pos)], device=env.device, dtype=env.dtype)  # 10 listener positions

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

    # Build each array along the local tangent so its look direction points to (0, 0).
    def build_array_positions(array_centre, offsets, invert_offsets=False):
        look_dir = -array_centre / np.linalg.norm(array_centre)  # from array centre to listener at origin
        tangent_dir = np.array([-look_dir[1], look_dir[0]], dtype=float)  # 90-degree rotation
        signed_offsets = -offsets if invert_offsets else offsets
        return [(array_centre + off * tangent_dir).astype(float).tolist() for off in signed_offsets]

    upper_array = build_array_positions(upper_centre, relative_offsets, invert_offsets=False)
    lower_array = build_array_positions(lower_centre, relative_offsets, invert_offsets=True)
    env.spk_pos = torch.tensor(upper_array + lower_array, device=env.device, dtype=env.dtype)
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
    training_stimulus_data_loader, stimulus_dataset = get_stim_loader(env.fs, env.stimulus_batch_size, 1, stimulus_type=env.selected_stim, device=env.device, dtype=env.dtype) # Get stimulus dataloader
    training_listener_data_loader, _ = get_listener_loader(env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling) # Get listener dataloader

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(fs=env.fs, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, sofa_file_path=env.sofa_path, device=env.device, dtype=env.dtype) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # Spectral Matching model
    unc_model = desena2020(loss = F.l1_loss, target = 0, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
    ipd_model = interaural_phase_difference_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # Interaural Phase Difference model
    regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)

    env.loc_weight = 1.0
    env.unc_weight = 1.0e1
    env.col_weight = 1e-01
    env.reg_weight = 1.0e-2
    env.flatness_weight = 1.0e-4

    listener_loss_kwargs = dict(device=env.device, dtype=env.dtype, listener_weights=listener_weights) if listener_weights is not None else dict(device=env.device, dtype=env.dtype)

    loss_dict = {
        "localisation": ListenerBasedLoss(model=loc_model, weight=env.loc_weight, **listener_loss_kwargs),
        # "uncertainty": ListenerBasedLoss(model=unc_model, weight=env.unc_weight, **listener_loss_kwargs),
        # "colouration": ListenerBasedLoss(model=col_model, weight=env.col_weight, **listener_loss_kwargs),
        # "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=env.reg_weight),
        # "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=env.reg_weight),
        # "regularisation_source_2": SourceBasedLoss(regulariser=regularisation, source_idx=2, weight=env.reg_weight),
        # "regularisation_source_3": SourceBasedLoss(regulariser=regularisation, source_idx=3, weight=env.reg_weight),
        "flatness_pair_0_1": SourceBasedLoss(
           regulariser=source_flatness(n_samples=env.n_samples, device=env.device, dtype=env.dtype),
           source_idx=(0, 1),
           weight=env.flatness_weight,
        ),
        "flatness_pair_2_3": SourceBasedLoss(
           regulariser=source_flatness(n_samples=env.n_samples, device=env.device, dtype=env.dtype),
           source_idx=(2, 3),
           weight=env.flatness_weight,
        ),
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
            branch.weight[centre_tap_idx] = 1.0

    # Tie mirrored loudspeakers between the two arrays to preserve symmetry.
    for local_idx in range(env.speakers_per_array):
        nn_model.branches[env.speakers_per_array + local_idx] = nn_model.branches[local_idx]
    
    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)
    
## Initliase saving folder and logger
    saver_obj = saver()
    logger = saver_obj if env.log else None

## Training loop
    # Train the neural network
    trainer = Train(n_epoch=env.epochs,
                    stim_loader = training_stimulus_data_loader,
                    lstn_loader = training_listener_data_loader,
                    nn=nn_model,
                    prop=stereo_prop,
                    loss_dict=loss_dict,
                    optimiser=nn_optimiser,
                    loggers = [logger, None]) # Initialize training class
    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

## Print and Plot results
    # Evaluate final neural network FIRs
    eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(env, eval_lstn_pos=env.lstn_pos)
    est_filters, estimation_dict, bin_stim = trainer.evaluate(eval_lstn_pos=eval_lstn_pos, eval_stimulus_data_loader=eval_stimulus_data_loader, loss_dict=eval_perceptual_dict)
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
        env=env,
    )
    plots = plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=bin_stim,
        lstn_pos=eval_lstn_pos,
        spk_pos=env.spk_pos,
        loss_dict=eval_perceptual_dict,
        target_pos=env.target_pos,
        fs=env.fs,
        stim = eval_stimulus_dataset.data,
        device=env.device,
        dtype=env.dtype,
        show=False,
    )

    fig_listener_weights, ax_listener_weights = plt.subplots(figsize=(6, 3))
    lstn_y = env.lstn_pos[:, 1].detach().cpu().numpy().astype(float, copy=False)
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
        spk_pos=env.spk_pos.detach().cpu().numpy(),
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
        spk_pos = env.spk_pos,
        lstn_pos = env.lstn_pos,
        loc = torch.as_tensor(estimation_dict["localisation"], device=env.device, dtype=env.dtype),
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
        fs=env.fs,
        lstn_pos=env.lstn_pos,
        spk_pos=[upper_centre.tolist(), lower_centre.tolist()],
        sofa_file_path=env.sofa_path,
        device=env.device,
        dtype=env.dtype,
    )
    loc_model.eval()
    with torch.no_grad():
        centre_stimulus = eval_stimulus_dataset.data[0:1].to(device=env.device, dtype=env.dtype)
        h_centre = torch.as_tensor(est_filters_centre, device=env.device, dtype=env.dtype).unsqueeze(0)
        bin_stim_centre = stereo_prop_centre(centre_stimulus, h_centre)
        loc_centre = loc_model(bin_stim_centre).squeeze(0)

    plots["arrows_and_directivity_centre"] = plot_arrows_with_directivity(
        est_filters=est_filters_centre,
        spk_pos=torch.as_tensor(spk_pos_centre, device=env.device, dtype=env.dtype),
        lstn_pos=env.lstn_pos,
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
