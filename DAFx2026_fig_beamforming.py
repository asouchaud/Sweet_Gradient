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


def localisation_error_metrics(target_angles, loc_before, loc_after, listener_weights):
    """
    Per-listener localisation error (estimated minus target azimuth, in degrees) before and
    after optimisation, reported both unweighted and weighted by the listener-position weights.
    """
    target = np.asarray(target_angles, dtype=float).ravel()
    before = np.asarray(loc_before, dtype=float).ravel()
    after = np.asarray(loc_after, dtype=float).ravel()
    weights = np.asarray(listener_weights, dtype=float).ravel()

    err_before = before - target
    err_after = after - target
    sum_w = float(np.sum(weights))

    def _wmean(values):
        return float(np.sum(weights * values) / sum_w) if sum_w > 0.0 else float(np.mean(values))

    return {
        "target_angle_deg": target.tolist(),
        "loc_before_deg": before.tolist(),
        "loc_after_deg": after.tolist(),
        "mae_before_deg": float(np.mean(np.abs(err_before))),
        "mae_after_deg": float(np.mean(np.abs(err_after))),
        "rmse_before_deg": float(np.sqrt(np.mean(err_before ** 2))),
        "rmse_after_deg": float(np.sqrt(np.mean(err_after ** 2))),
        "weighted_mae_before_deg": _wmean(np.abs(err_before)),
        "weighted_mae_after_deg": _wmean(np.abs(err_after)),
        "weighted_rmse_before_deg": float(np.sqrt(_wmean(err_before ** 2))),
        "weighted_rmse_after_deg": float(np.sqrt(_wmean(err_after ** 2))),
    }


def compute_group_directivity_db(
    filters, spk_pos_np, group_indices, fs, freqs_eval_hz, radius, n_angles=360, c=343.0
):
    """
    Frequency-averaged far-field directivity (in dB) of a loudspeaker group as a function of
    observation angle around the group centroid. Mirrors ``plot_arrows_with_directivity`` with
    ``directivity_mode='avg'`` so the returned curve matches the plotted blue curve.
    """
    group_indices = np.asarray(group_indices, dtype=int).reshape(-1)
    h_group = np.atleast_2d(np.asarray(filters, dtype=float))[group_indices]
    p_group = np.asarray(spk_pos_np, dtype=float)[group_indices]
    group_mid = p_group.mean(axis=0)

    theta = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    n_taps = h_group.shape[1]
    n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    omega = 2 * np.pi * freqs
    H_group = np.fft.rfft(h_group, n=n_fft, axis=1)
    freq_bin_idx = [int(np.argmin(np.abs(freqs - f))) for f in freqs_eval_hz]

    mag2 = np.zeros((len(freqs_eval_hz), n_angles))
    for k, ang in enumerate(theta):
        obs = group_mid + np.array([radius * np.cos(ang), radius * np.sin(ang)])
        tau = np.linalg.norm(obs - p_group, axis=1) / c
        phase = np.exp(-1j * np.outer(tau, omega))
        H_sum = np.sum(H_group * phase, axis=0)
        for fi, bi in enumerate(freq_bin_idx):
            mag2[fi, k] = np.abs(H_sum[bi]) ** 2

    dir_avg_db = 10.0 * np.log10(np.mean(mag2, axis=0) + 1e-12)
    return theta, dir_avg_db, group_mid


def _db_at_nearest_angle_deg(sample_angles_deg, levels_db, target_angle_deg):
    sample_angles_deg = np.asarray(sample_angles_deg, dtype=np.float64).ravel()
    levels_db = np.asarray(levels_db, dtype=np.float64).ravel()
    target_wrapped = float(np.mod(target_angle_deg, 360.0))
    samples_wrapped = np.mod(sample_angles_deg, 360.0)
    separation_deg = np.abs((samples_wrapped - target_wrapped + 180.0) % 360.0 - 180.0)
    return float(levels_db[int(np.argmin(separation_deg))])


def directivity_match_metrics(theta, dir_avg_db, group_mid, overlay_curves, overlay_labels=None):
    """
    Compare the estimated group directivity against each Rodenas overlay curve after
    normalisation. Both curves are aligned to 0 dB at the array steering direction (the wedge
    mid-angle) exactly as in the figure, then the overlay is interpolated onto the estimated
    angle grid and dB error statistics are computed over the overlapping angular range.
    """
    cy = float(group_mid[1])
    if cy >= 0.0:
        theta_wedge0 = np.pi
        theta_wedge1 = np.pi + np.pi / 3.0
    else:
        theta_wedge0 = np.pi - np.pi / 3.0
        theta_wedge1 = np.pi
    mid_rad = 0.5 * (theta_wedge0 + theta_wedge1)
    mid_angle_deg = float(np.rad2deg(mid_rad) % 360.0)

    delta_theta = np.abs((theta - mid_rad + np.pi) % (2 * np.pi) - np.pi)
    k_mid = int(np.argmin(delta_theta))
    dir_est_aligned = dir_avg_db - dir_avg_db[k_mid]

    theta_deg = np.rad2deg(theta)
    wedge0_deg = float(np.rad2deg(theta_wedge0))
    wedge1_deg = float(np.rad2deg(theta_wedge1))
    theta_wrapped = np.mod(theta, 2 * np.pi)
    mask = (theta_wrapped >= theta_wedge0) & (theta_wrapped <= theta_wedge1)
    theta_sel_deg = np.mod(theta_deg[mask], 360.0)
    est_sel = dir_est_aligned[mask]

    overlay_curves = [np.asarray(curve, dtype=float) for curve in overlay_curves]
    overlay_angle_offset_deg = 180.0 - float(np.min(np.mod(overlay_curves[0][:, 0], 360.0)))

    metrics = []
    for ci, curve in enumerate(overlay_curves):
        label = overlay_labels[ci] if (overlay_labels and ci < len(overlay_labels)) else f"overlay_{ci}"

        angles_deg_plot = np.mod(curve[:, 0] + overlay_angle_offset_deg, 360.0)
        db_sel = np.flip(curve[:, 1])
        if cy < 0.0:
            angles_deg_plot = np.mod(360.0 - angles_deg_plot, 360.0)
        db_at_mid = _db_at_nearest_angle_deg(angles_deg_plot, db_sel, mid_angle_deg)
        db_aligned = db_sel - db_at_mid

        order = np.argsort(angles_deg_plot)
        ov_ang = angles_deg_plot[order]
        ov_db = db_aligned[order]

        lo = max(wedge0_deg, float(ov_ang.min()))
        hi = min(wedge1_deg, float(ov_ang.max()))
        in_range = (theta_sel_deg >= lo) & (theta_sel_deg <= hi)
        if not np.any(in_range):
            metrics.append({"label": label, "rmse_db": float("nan"), "mae_db": float("nan"),
                            "max_abs_db": float("nan"), "n_points": 0, "angle_range_deg": [lo, hi]})
            continue

        ov_interp = np.interp(theta_sel_deg[in_range], ov_ang, ov_db)
        diff = est_sel[in_range] - ov_interp
        metrics.append({
            "label": label,
            "rmse_db": float(np.sqrt(np.mean(diff ** 2))),
            "mae_db": float(np.mean(np.abs(diff))),
            "max_abs_db": float(np.max(np.abs(diff))),
            "n_points": int(np.sum(in_range)),
            "angle_range_deg": [lo, hi],
        })
    return metrics


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=200,
        n_samples=40,
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

    d = env.speaker_distance * np.cos(np.deg2rad(env.array_centre_angles_deg[0]))
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

    # Place the speakers along a line on the y-axis, comment out the "turning" code.
    def build_array_positions(array_centre, offsets, invert_offsets=False):
       # look_dir = -array_centre / np.linalg.norm(array_centre)  # from array centre to listener at origin
       # tangent_dir = np.array([-look_dir[1], look_dir[0]], dtype=float)  # 90-degree rotation
       signed_offsets = -offsets if invert_offsets else offsets
       # return [(array_centre + off * tangent_dir).astype(float).tolist() for off in signed_offsets]
       # No turning; speakers are placed along the global y-axis direction
       return [(array_centre[0], array_centre[1] + off) for off in signed_offsets]

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
        # "flatness_pair_0_1": SourceBasedLoss(
        #    regulariser=source_flatness(n_samples=env.n_samples, 
        #                                 device=env.device, 
        #                                 dtype=env.dtype,
        #                                 spk_pos=env.spk_pos[[0, 1]],
        #                                 steer_point=[0.0, 0.0],
        #                                 fs = env.fs,
        #                                 ),
        #    source_idx=(0, 1),
        #    weight=env.flatness_weight,
        # ),
        # "flatness_pair_2_3": SourceBasedLoss(
        #    regulariser=source_flatness(n_samples=env.n_samples, 
        #                                device=env.device, 
        #                                dtype=env.dtype,
        #                                spk_pos=env.spk_pos[[2, 3]],
        #                                steer_point=[0.0, 0.0],
        #                                fs = env.fs,
        #                                ),
        #    source_idx=(2, 3),
        #    weight=env.flatness_weight,
        # )
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
                    logger=logger) # Initialize training class

    # Build evaluation objects up-front so we can snapshot localisation before optimising.
    eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(env, eval_lstn_pos=env.lstn_pos)

    # Snapshot pre-optimisation localisation using the initial (delta) FIRs.
    _, initial_estimation_dict, _ = trainer.evaluate(eval_lstn_pos=eval_lstn_pos, eval_stimulus_data_loader=eval_stimulus_data_loader, loss_dict=eval_perceptual_dict)

    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

## Print and Plot results
    # Evaluate final neural network FIRs
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

## Quantitative analysis of the results
    import json

    # Per-listener target azimuth used by the localisation model (degrees relative to +x axis).
    target_angles_deg = eval_perceptual_dict["localisation"].model.target.detach().cpu().numpy()

    loc_metrics = localisation_error_metrics(
        target_angles=target_angles_deg,
        loc_before=initial_estimation_dict["localisation"],
        loc_after=estimation_dict["localisation"],
        listener_weights=listener_weights,
    )

    # Directivity of the (right) optimised loudspeaker array vs. the Rodenas references.
    dir_theta, dir_avg_db, dir_group_mid = compute_group_directivity_db(
        filters=est_filters,
        spk_pos_np=env.spk_pos.detach().cpu().numpy(),
        group_indices=right_array_indices,
        fs=env.fs,
        freqs_eval_hz=env.directivity_freqs_hz,
        radius=env.directivity_radius_m,
        n_angles=360,
    )
    directivity_metrics = directivity_match_metrics(
        theta=dir_theta,
        dir_avg_db=dir_avg_db,
        group_mid=dir_group_mid,
        overlay_curves=[rodenas2003_fig12_opt, rodenas2003_fig12_1000Hz],
        overlay_labels=["rodenas2003 (optimum)", "rodenas2003 (1000 Hz)"],
    )

    analysis = {"localisation": loc_metrics, "directivity_match": directivity_metrics}

    print("\n================ Results analysis ================")
    print("Localisation error vs target angle (degrees):")
    print(f"  Unweighted MAE : {loc_metrics['mae_before_deg']:.2f} (before) -> {loc_metrics['mae_after_deg']:.2f} (after)")
    print(f"  Unweighted RMSE: {loc_metrics['rmse_before_deg']:.2f} (before) -> {loc_metrics['rmse_after_deg']:.2f} (after)")
    print(f"  Weighted   MAE : {loc_metrics['weighted_mae_before_deg']:.2f} (before) -> {loc_metrics['weighted_mae_after_deg']:.2f} (after)")
    print(f"  Weighted   RMSE: {loc_metrics['weighted_rmse_before_deg']:.2f} (before) -> {loc_metrics['weighted_rmse_after_deg']:.2f} (after)")
    print("Directivity match vs Rodenas (after normalisation, within steering wedge):")
    for m in directivity_metrics:
        print(f"  {m['label']}: RMSE {m['rmse_db']:.2f} dB, MAE {m['mae_db']:.2f} dB, max |err| {m['max_abs_db']:.2f} dB (n={m['n_points']})")
    print("==================================================\n")

    with open(saver_obj.run_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=4)

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
