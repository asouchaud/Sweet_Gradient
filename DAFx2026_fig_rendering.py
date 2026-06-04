# Enable MPS fallback for PyTorch on macOS
import os
import json
import sofar as sf
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import interface libraries
from pathlib import Path
import time
from types import SimpleNamespace

# Import computational libraries
import numpy as np
import torch
import torch.nn.functional as F

# Import Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors


# Import project modules
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.plotting.results_plot import plot_results
from scripts.setup.setup import set_seed, build_eval_objects
from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.datasets.listener_dataset import get_listener_loader
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
from scripts.saving.save_results import saver


def compute_optimisation_analysis(estimation_dict, eval_perceptual_dict, eval_lstn_pos, env):
    """
    Summarise post-optimisation perceptual metrics at the evaluation listeners.

    Localisation errors are computed against the per-listener target azimuth from the
    llado2025 model (not the nominal ``target_angle`` alone). Uncertainty and colouration
    are reported relative to a target of zero.
    """
    centre_idx = 0
    lstn_y = eval_lstn_pos.detach().cpu().numpy()[:, 1].tolist()

    analysis = {
        "run_label": getattr(env, "run_label", None),
        "plot_row_title": getattr(env, "plot_row_title", None),
        "target_angle_deg_nominal": float(env.target_angle),
        "eval_listener_y_m": lstn_y,
        "training_time_s": float(getattr(env, "time_train_seconds", np.nan)),
        "notes": {
            "localisation": "Signed error = estimated minus per-listener target azimuth [deg].",
            "uncertainty": "De Sena et al. uncertainty metric; target is 0 (lower is better).",
            "colouration": (
                "Log spectral distance (LSD) in dB from spectral_matching; "
                "normalised values divide by the model loss_norm used in training."
            ),
        },
    }

    if "localisation" in estimation_dict and "localisation" in eval_perceptual_dict:
        target = np.asarray(
            eval_perceptual_dict["localisation"].model.target.detach().cpu().numpy(),
            dtype=float,
        ).ravel()
        estimated = np.asarray(estimation_dict["localisation"], dtype=float).ravel()
        signed_err = estimated - target
        abs_err = np.abs(signed_err)

        analysis["localisation"] = {
            "target_angle_deg": target.tolist(),
            "estimated_angle_deg": estimated.tolist(),
            "signed_error_deg": signed_err.tolist(),
            "abs_error_deg": abs_err.tolist(),
            "mae_deg": float(np.mean(abs_err)),
            "rmse_deg": float(np.sqrt(np.mean(signed_err ** 2))),
            "max_abs_error_deg": float(np.max(abs_err)),
            "centre_listener": {
                "y_m": float(lstn_y[centre_idx]),
                "target_angle_deg": float(target[centre_idx]),
                "estimated_angle_deg": float(estimated[centre_idx]),
                "signed_error_deg": float(signed_err[centre_idx]),
                "abs_error_deg": float(abs_err[centre_idx]),
            },
        }

    for key, target_ref in (("uncertainty", 0.0), ("colouration", 0.0)):
        if key not in estimation_dict:
            continue
        values = np.asarray(estimation_dict[key], dtype=float).ravel()
        abs_err = np.abs(values - target_ref)
        block = {
            "per_listener": values.tolist(),
            "mean": float(np.mean(values)),
            "mae": float(np.mean(abs_err)),
            "max": float(np.max(values)),
            "centre_listener": {
                "y_m": float(lstn_y[centre_idx]),
                "value": float(values[centre_idx]),
            },
        }
        if key in eval_perceptual_dict:
            loss_norm = float(eval_perceptual_dict[key].model.loss_norm)
            block["loss_norm"] = loss_norm
            block["mean_loss_normalised"] = float(np.mean(values) / loss_norm)
            block["centre_listener"]["loss_normalised"] = float(values[centre_idx] / loss_norm)
        analysis[key] = block

    return analysis


def print_optimisation_analysis(analysis, run_label=None):
    label = run_label or analysis.get("run_label") or analysis.get("plot_row_title") or "run"
    print(f"\n================ Optimisation analysis ({label}) ================")
    print(
        f"Nominal target angle: {analysis['target_angle_deg_nominal']:.2f} deg  |  "
        f"Training time: {analysis['training_time_s']:.2f} s"
    )

    loc = analysis.get("localisation")
    if loc is not None:
        print("Localisation (vs per-listener target):")
        print(f"  Mean |error| (MAE) : {loc['mae_deg']:.3f} deg")
        print(f"  RMSE               : {loc['rmse_deg']:.3f} deg")
        print(f"  Max |error|        : {loc['max_abs_error_deg']:.3f} deg")
        centre = loc["centre_listener"]
        print(
            f"  Centre listener    : est {centre['estimated_angle_deg']:.2f} deg, "
            f"target {centre['target_angle_deg']:.2f} deg, "
            f"|error| {centre['abs_error_deg']:.3f} deg"
        )

    unc = analysis.get("uncertainty")
    if unc is not None:
        print("Uncertainty (target = 0):")
        print(f"  Mean               : {unc['mean']:.4f}")
        print(f"  Mean |value| (MAE) : {unc['mae']:.4f}")
        print(f"  Centre listener    : {unc['centre_listener']['value']:.4f}")

    col = analysis.get("colouration")
    if col is not None:
        print("Colouration / LSD (target = 0, lower is better):")
        print(f"  Mean LSD           : {col['mean']:.3f} dB")
        print(f"  Mean |LSD| (MAE)   : {col['mae']:.3f} dB")
        if "mean_loss_normalised" in col:
            print(f"  Mean normalised    : {col['mean_loss_normalised']:.4f}")
        print(f"  Centre listener    : {col['centre_listener']['value']:.3f} dB")

    print("Per-listener breakdown (y [m] -> loc |error| [deg], uncertainty, LSD [dB]):")
    n_listeners = len(analysis["eval_listener_y_m"])
    for i in range(n_listeners):
        y = analysis["eval_listener_y_m"][i]
        parts = [f"  y={y:+.3f} m:"]
        if loc is not None:
            parts.append(f"|loc err|={loc['abs_error_deg'][i]:.3f} deg")
        if unc is not None:
            parts.append(f"unc={unc['per_listener'][i]:.4f}")
        if col is not None:
            parts.append(f"LSD={col['per_listener'][i]:.3f} dB")
        print("  ".join(parts))
    print("=" * 66)


def print_cross_run_analysis(analyses):
    """Print a compact table comparing key metrics across optimisation runs."""
    if not analyses:
        return

    headers = [
        "Run",
        "|loc err| [deg]",
        "RMSE loc [deg]",
        "Uncertainty",
        "LSD [dB]",
        "Col. norm.",
        "Train [s]",
    ]
    rows = []
    for analysis in analyses:
        label = analysis.get("plot_row_title") or analysis.get("run_label") or "?"
        loc = analysis.get("localisation", {})
        unc = analysis.get("uncertainty", {})
        col = analysis.get("colouration", {})
        rows.append([
            label,
            f"{loc.get('mae_deg', float('nan')):.3f}",
            f"{loc.get('rmse_deg', float('nan')):.3f}",
            f"{unc.get('mean', float('nan')):.4f}",
            f"{col.get('mean', float('nan')):.3f}",
            f"{col.get('mean_loss_normalised', float('nan')):.4f}",
            f"{analysis.get('training_time_s', float('nan')):.1f}",
        ])

    col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:{w}s}}" for w in col_widths)

    print(f"\n{'=' * 72}\nCross-run optimisation summary (mean over evaluation listeners)\n{'=' * 72}")
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in col_widths]))
    for row in rows:
        print(fmt.format(*row))
    print("=" * 72)


def run_experiment(loss_dict_builder, run_label, plot_row_title=None):
    repo_root = Path(__file__).resolve().parent
    d = 60
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=300,
        num_lstn_pos=30,
        listener_batch_size=5,  # 5 mirrored pairs plus the centre listener
        symmetric_listener_sampling=True,
        n_samples=64,
        target_angle=10.0,
        selected_stim="impulse",
        stimulus_batch_size=1,
        n_stimulus=1,
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        lr=0.01,
        seed=46,
        log=1,
        fs=48000,
    )
    env.plot_row_title = plot_row_title if plot_row_title is not None else run_label
    env.run_label = run_label
    env.target_pos=[d, float(np.tan(np.deg2rad(env.target_angle)) * d)],
    env.dtype = torch.float32
    # env.device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env.device_str = "cpu"
    env.device = torch.device(env.device_str)

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    n_per_segment = env.num_lstn_pos // 4
    segments = [
        np.linspace(-0.5, -0.325, n_per_segment),
        np.linspace(-0.225, -0.05, n_per_segment),
        np.linspace(0.05, 0.225, n_per_segment),
        np.linspace(0.325, 0.5, n_per_segment),
    ]
    y_positions = np.concatenate(segments)
    env.lstn_pos = [[0.0, float(y)] for y in y_positions]
    # env.lstn_pos = [[0.0, float(y)] for y in np.linspace(-0.5, 0.5, env.num_lstn_pos)] 
    base_angle_rad = np.deg2rad(30.0)
    env.speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
    env.spk_pos = [
        [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
        [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
    ]
    env.n_sources = len(env.spk_pos)
    env.lstn_pos = torch.tensor(env.lstn_pos, device=env.device, dtype=env.dtype)
    env.spk_pos = torch.tensor(env.spk_pos, device=env.device, dtype=env.dtype)

    # Reset the seed so each run starts from the same point.
    set_seed(seed=env.seed)

## Load data
    training_stimulus_data_loader, stimulus_dataset = get_stim_loader(
        env.fs,
        env.stimulus_batch_size,
        env.n_stimulus,
        stimulus_type=env.selected_stim,
        device=env.device,
        dtype=env.dtype,
    )
    training_listener_data_loader, _ = get_listener_loader(
        env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling
    )

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(
        fs=env.fs,
        lstn_pos=env.lstn_pos,
        spk_pos=env.spk_pos,
        sofa_file_path=env.sofa_path,
        device=env.device,
        dtype=env.dtype,
    )

    # Initiate perceptual models
    loc_model = llado2025(
        loss=F.mse_loss,
        target_pos=env.target_pos,
        fs=env.fs,
        sofa_path=env.sofa_path,
        lstn_pos=env.lstn_pos,
        spk_pos=env.spk_pos,
        balance_interaural_cue_grads=False,
        device=env.device,
        dtype=env.dtype,
    )
    col_model = spectral_matching(
        loss=F.l1_loss,
        target=0,
        target_pos=env.target_pos,
        fs=env.fs,
        sofa_path=env.sofa_path,
        lstn_pos=env.lstn_pos,
        device=env.device,
        dtype=env.dtype,
    )
    unc_model = desena2020(
        loss=F.mse_loss,
        target=0,
        fs=env.fs,
        sofa_path=env.sofa_path,
        lstn_pos=env.lstn_pos,
        device=env.device,
        dtype=env.dtype,
    )
    regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)
    loss_dict = loss_dict_builder(loc_model, col_model, unc_model, regularisation, env.device, env.dtype)

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

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

## Initliase saving folder and logger
    saver_obj = saver()
    logger = saver_obj if env.log else None

## Training loop
    # Train the neural network
    trainer = Train(
        n_epoch=env.epochs,
        stim_loader=training_stimulus_data_loader,
        lstn_loader=training_listener_data_loader,
        nn=nn_model,
        prop=stereo_prop,
        loss_dict=loss_dict,
        optimiser=nn_optimiser,
        logger=logger,
    )
    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

## Print and Plot results
    # Evaluate final neural network FIRs
    eval_lstn_pos = torch.tensor(
        [[0.0, 0.0], [0.0, 0.275], [0.0, -0.275]],
        device=env.device,
        dtype=env.dtype,
    )
    eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(
        env,
        eval_lstn_pos=eval_lstn_pos,
        n_eval_stim=env.n_stimulus,
    )
    trainer.prop = stereo_prop
    est_filters, estimation_dict, bin_stim = trainer.evaluate(
        eval_lstn_pos=eval_lstn_pos,
        eval_stimulus_data_loader=eval_stimulus_data_loader,
        loss_dict=eval_perceptual_dict,
    )
    # Normalise filters to prevent clipping 
    est_filters /= np.max(np.abs(est_filters)) / 0.9 # TODO: move this to a more appropriate place at the end of the training loop

    analysis = compute_optimisation_analysis(
        estimation_dict=estimation_dict,
        eval_perceptual_dict=eval_perceptual_dict,
        eval_lstn_pos=eval_lstn_pos,
        env=env,
    )
    print_optimisation_analysis(analysis, run_label=run_label)

    saver_obj.save_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        fs=env.fs,
        env=env,
        loss_dict=loss_dict,
    )
    with open(saver_obj.run_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=4)
    artifacts_dir = saver_obj.run_dir / "artifacts"
    stim_to_save = eval_stimulus_dataset.data.detach().cpu().numpy()
    if stim_to_save.ndim == 3:
        stim_to_save = stim_to_save[0, 0]
    np.save(artifacts_dir / "stimulus.npy", stim_to_save)
    plots = plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=np.expand_dims(bin_stim, axis=0),
        lstn_pos=eval_lstn_pos,
        spk_pos=env.spk_pos,
        loss_dict=eval_perceptual_dict,
        target_pos=env.target_pos,
        fs=env.fs,
        stim=eval_stimulus_dataset.data,
        device=env.device,
        dtype=env.dtype,
        show=False,
    )
    saver_obj.save_plots(plots)

    # Save per-epoch source FIR evolution as videos (one per source)
    # saver_obj.save_filter_video(trainer.h_history, source_index=0, fs=env.fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=1, fs=env.fs)

    plt.close('all')

    return (
        saver_obj.run_dir,
        eval_lstn_pos.detach().cpu().numpy(),
        env.spk_pos.detach().cpu().numpy(),
        env.plot_row_title,
        analysis,
    )

def causal_conv1d(x, kernel):
    """
    x: (T,) input signal
    kernel: (K,) convolution kernel
    returns: (T,) causal convolution output
    """
    
    K = len(kernel)

    # pad on the left so output length matches input
    x_padded = np.pad(x, (K-1, 0), mode='constant')

    y = np.zeros_like(x, dtype=float)

    for t in range(len(x)):
        y[t] = np.sum(x_padded[t:t+K] * kernel[::-1])

    return y

def smooth_third_octave(freq, mag_db):
    smoothed = np.zeros_like(mag_db)

    k = 2**(1/6)  # 1/3 octave half-width

    for i, fc in enumerate(freq):
        f_low = fc / k
        f_high = fc * k

        idx = np.where((freq >= f_low) & (freq <= f_high))[0]

        smoothed[i] = np.mean(mag_db[idx])

    return smoothed

def sofa_find(source_angle, sofaile):
    pos = sofaile.SourcePosition[:, :2]          # (N,2)

    source = np.array([[source_angle, 0]])       # (1,2)

    dists = np.linalg.norm(source - pos, axis=1)

    return np.argmin(dists)


def _ear_colouration_spectra_db(
    stimulus,
    filters,
    lstn_pos,
    spk_pos,
    ref_hrir,
    sofa_path,
    ear_idx=0,
    fs=48000,
    n_fft=4096,
):
    """Smoothed magnitude spectra [dB] for reference and evaluation listeners (ear_idx: 0=left, 1=right)."""
    if ear_idx not in (0, 1):
        raise ValueError("ear_idx must be 0 (left) or 1 (right)")

    stim_torch = torch.tensor(stimulus, dtype=torch.float32).view(1, 1, -1)
    prop = BinauralFreeFieldPropagation(
        fs=fs,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        sofa_file_path=sofa_path,
        dtype=torch.float32,
    )
    filters_torch = torch.tensor(filters, dtype=torch.float32).unsqueeze(0)
    listener_idx = torch.arange(len(lstn_pos))
    bin_stim = prop(stim_torch, filters_torch, listener_idx).numpy()
    if bin_stim.ndim == 4:
        bin_stim = bin_stim[0]

    freq = np.fft.rfftfreq(n=n_fft, d=1 / fs)
    bin_stim_fft = 20 * np.log10(
        np.abs(np.fft.rfft(bin_stim, n=n_fft, axis=-1)) + 1e-12
    )

    ref_stim = causal_conv1d(stimulus, ref_hrir[ear_idx])
    ref_fft = 20 * np.log10(np.abs(np.fft.rfft(ref_stim, n=n_fft)) + 1e-12)

    ref_fft_s = smooth_third_octave(freq, ref_fft)
    n_listeners = len(lstn_pos)
    bin_fft_s = np.zeros((n_listeners, len(freq)), dtype=float)
    for j in range(n_listeners):
        bin_fft_s[j] = smooth_third_octave(freq, bin_stim_fft[j, ear_idx, :])

    return freq, ref_fft_s, bin_fft_s


def DAFx2026_fig5_plots(
    run_folders,
    lstn_pos,
    spk_pos,
    row_titles=None,
    ear_idx=0,
    output_pdf="DAFx2026_fig5.pdf",
    show=True,
):

    repo_root = Path(__file__).resolve().parent
    sofa_path = str(repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa")
    sofaile = sf.read_sofa(sofa_path)
    idx = sofa_find(0, sofaile)
    ref_hrir = sofaile.Data_IR[idx]

    fig_colors = [ "#33a02c", "#1f78b4", "#e31a1c"]

    arrow_width = 3
    arrow_head_width = 0.12
    arrow_head_len = 0.14
    arrow_len = 1.2
    ref_mag_width = 5
    est_mag_width = 3
    left_ir_width = 3
    right_ir_width = 2 
    fontsize = 20

    n_rows = len(run_folders)
    if row_titles is None:
        row_titles = []
        for folder_name in run_folders:
            config_path = os.path.join(folder_name, "config.json")
            with open(config_path) as f:
                cfg = json.load(f)
            row_titles.append(cfg.get("main", {}).get("plot_row_title", cfg.get("run_name", "")))
    elif len(row_titles) != n_rows:
        raise ValueError("row_titles must have the same length as run_folders")

    lsd_observed = []
    for folder_name in run_folders:
        with open(os.path.join(folder_name, "results.json")) as f:
            lsd_observed.extend(np.asarray(json.load(f)["colouration"]).ravel().tolist())
    lsd_observed = np.asarray(lsd_observed, dtype=float)
    lsd_step = 0.5
    vmin_lsd = float(np.floor(np.min(lsd_observed) / lsd_step) * lsd_step)
    vmax_lsd = float(np.ceil(np.max(lsd_observed) / lsd_step) * lsd_step)
    if vmax_lsd <= vmin_lsd:
        vmax_lsd = vmin_lsd + lsd_step

    norm = colors.Normalize(vmin=vmin_lsd, vmax=vmax_lsd)
    cmap = colors.LinearSegmentedColormap.from_list(
        "bright_green_red",
        [(0,1,0), (0.8,0.4,0), (0.8,0,0)]  # green → yellow → red
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    freq_plot_start = 9
    mag_db_values = []
    for folder_name in run_folders:
        filters_path = os.path.join(folder_name, "artifacts", "est_filters.npy")
        stimulus_path = os.path.join(folder_name, "artifacts", "stimulus.npy")
        filters = np.load(filters_path)
        filters = filters * 0.9 / np.max(np.abs(filters))
        stimulus = np.load(stimulus_path)
        freq, ref_fft_s, bin_fft_s = _ear_colouration_spectra_db(
            stimulus=stimulus,
            filters=filters,
            lstn_pos=lstn_pos,
            spk_pos=spk_pos,
            ref_hrir=ref_hrir,
            sofa_path=sofa_path,
            ear_idx=ear_idx,
        )
        mag_db_values.append(ref_fft_s[freq_plot_start:])
        for j in range(len(lstn_pos)):
            mag_db_values.append(bin_fft_s[j, freq_plot_start:])
    mag_db_all = np.concatenate(mag_db_values)
    mag_db_all = mag_db_all[np.isfinite(mag_db_all)]
    ymin_mag = float(np.min(mag_db_all))
    ymax_mag = float(np.max(mag_db_all))

    fig = plt.figure(figsize=(10.5, 3.15 * n_rows + 1.1))
    top_gs, bottom_u, bottom_l = 0.90, 0.10, 0.38
    axes_frac_u = top_gs - bottom_u
    axes_frac_l = top_gs - bottom_l
    last_row_ratio = axes_frac_u / axes_frac_l
    height_ratios = [1.0] * (n_rows - 1) + [last_row_ratio] if n_rows > 1 else [1.0]
    subfigs = fig.subfigures(n_rows, 1, squeeze=False, hspace=0.01, height_ratios=height_ratios)
    axes_data = np.empty((n_rows, 3), dtype=object)
    for r in range(n_rows):
        row_subfig = subfigs[r, 0]
        row_subfig.suptitle(row_titles[r], fontsize=fontsize, y=1.02, va="top")
        bottom_pad = bottom_l if r == n_rows - 1 else bottom_u
        axs = row_subfig.subplots(
            1,
            3,
            gridspec_kw={"wspace": 0.42, "left": 0.07, "right": 0.995, "top": top_gs, "bottom": bottom_pad},
        )
        for c in range(3):
            axes_data[r, c] = axs[c]

    target_angle_by_folder = {}
    for r in range(n_rows):
        folder_name = run_folders[r]
        json_file_path = os.path.join(folder_name, "results.json")
        with open(json_file_path) as f:
            json_data = json.load(f)
        if folder_name not in target_angle_by_folder:
            config_path = os.path.join(folder_name, "config.json")
            with open(config_path) as f:
                cfg = json.load(f)
            main = cfg.get("main", {})
            try:
                target_angle_by_folder[folder_name] = float(main["target_angle"])
            except (KeyError, TypeError, ValueError):
                tp = main.get("target_pos")
                if tp is not None and len(tp) > 0:
                    row = tp[0] if isinstance(tp[0], (list, tuple)) else tp
                    target_angle_by_folder[folder_name] = float(
                        np.rad2deg(np.arctan2(float(row[1]), float(row[0])))
                    )
                else:
                    target_angle_by_folder[folder_name] = 0.0
        target_angle_deg = target_angle_by_folder[folder_name]

        filters_path = os.path.join(folder_name, "artifacts", "est_filters.npy")
        filters = np.load(filters_path)
        filters = filters * 0.9 / np.max(np.abs(filters))

        is_bottom_row = r == n_rows - 1

        for c in range(3):
            ax = axes_data[r, c]

            #### IRs PLOT ####
            if c == 0:
                ax.plot(filters[0], linewidth = left_ir_width)
                ax.plot(filters[1], linewidth = right_ir_width)
                ax.grid()
                ax.set_ylim(-1,1)
                ax.set_ylabel("Amplitude", fontsize = fontsize)

                if is_bottom_row:
                    ax.set_xlabel("Time index", fontsize = fontsize)
                else:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            #### COLOURATION PLOT ####
            if c == 1:
                stimulus_path = os.path.join(folder_name, "artifacts", "stimulus.npy")
                stimulus = np.load(stimulus_path)

                freq, ref_fft_s, bin_fft_s = _ear_colouration_spectra_db(
                    stimulus=stimulus,
                    filters=filters,
                    lstn_pos=lstn_pos,
                    spk_pos=spk_pos,
                    ref_hrir=ref_hrir,
                    sofa_path=sofa_path,
                    ear_idx=ear_idx,
                )

                for j in range(len(lstn_pos)):
                    ax.semilogx(freq[freq_plot_start:], bin_fft_s[j, freq_plot_start:], color=fig_colors[j], linewidth=est_mag_width)
                ax.semilogx(freq[freq_plot_start:], ref_fft_s[freq_plot_start:], color="#000000", linewidth=ref_mag_width)
                ax.grid()
                ax.set_xlim(80, 20000)
                ax.set_ylim(ymin_mag, ymax_mag)
                ax.set_ylabel("Magnitude [dB]", fontsize = fontsize)
                if not is_bottom_row:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    ax.set_xlabel("Frequency [Hz]", fontsize = fontsize)

            #### ARROWS PLOT #####
            if c == 2:
                loc = json_data['localisation']
                unc = json_data['uncertainty']
                col = json_data['colouration']

                ax.scatter(spk_pos[:,0], spk_pos[:,1], c='black', label='Speaker', marker = 'x')
                ax.scatter(lstn_pos[:,0], lstn_pos[:,1], c='blue', label='Listener positions')

                theta_rad = np.deg2rad(target_angle_deg)
                c_dir, s_dir = np.cos(theta_rad), np.sin(theta_rad)
                t_ray = np.linspace(-2.5, 2.5, 200)
                for x, y, ang, leng, col_val in zip(lstn_pos[:,0], lstn_pos[:,1], loc, unc, col):
                    x_lin = x + t_ray * c_dir
                    y_lin = y + t_ray * s_dir
                    ax.plot(x_lin, y_lin, color="#000000", linestyle="dashed", lw=1, zorder=2)

                    dx = (1-leng)*arrow_len * np.cos(np.deg2rad(ang))
                    dy = (1-leng)*arrow_len * np.sin(np.deg2rad(ang))

                    color = cmap(norm(col_val))

                    ax.arrow(float(x), float(y), float(dx), float(dy),
                            head_width=arrow_head_width, head_length=arrow_head_len, fc=color, ec=color, linewidth = arrow_width, zorder = 3)

                ax.set_ylim(-1.25,1.25)
                ax.set_xlim(-0.5,2.0)
                ax.grid(zorder = 1)
                ax.set_ylabel('Y position [m]', fontsize = fontsize)
                if not is_bottom_row:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    ax.set_xlabel('X position [m]', fontsize = fontsize)

            ax.tick_params(axis='both', labelsize=fontsize)

    fig.subplots_adjust(bottom=0.36, top=0.99, left=0.06, right=0.99)
    cbar_ax = fig.add_axes([0.1, 0.006, 0.8, 0.024])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_xlabel("LSD [dB] (lower the better)", fontsize=fontsize, labelpad=5)
    repo_root = Path(__file__).resolve().parent
    out_path = repo_root / output_pdf if not os.path.isabs(output_pdf) else Path(output_pdf)
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.38)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def regenerate_fig5_from_latest_runs(
    results_dir=None,
    ear_idx=0,
    output_pdf=None,
    show=False,
):
    """Rebuild a fig. 5 PDF from the four most recent rendering result folders."""
    if output_pdf is None:
        output_pdf = "DAFx2026_fig5_left_ear.pdf" if ear_idx == 0 else "DAFx2026_fig5_right_ear.pdf"

    repo_root = Path(__file__).resolve().parent
    results_dir = Path(results_dir) if results_dir is not None else repo_root / "results"
    run_dirs = sorted(
        results_dir.glob("DAFx2026_fig_rendering_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(run_dirs) < 4:
        raise FileNotFoundError(f"Need at least 4 result folders in {results_dir}")

    folders = list(reversed(run_dirs[:4]))
    cfg0 = json.loads((folders[0] / "config.json").read_text(encoding="utf-8"))
    spk_pos = np.asarray(cfg0["main"]["spk_pos"], dtype=float)
    lstn_pos = np.array([[0.0, 0.0], [0.0, 0.275], [0.0, -0.275]], dtype=float)
    row_titles = []
    for folder in folders:
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        row_titles.append(cfg.get("main", {}).get("plot_row_title", cfg.get("run_name", folder.name)))

    return DAFx2026_fig5_plots(
        folders,
        lstn_pos,
        spk_pos,
        row_titles=row_titles,
        ear_idx=ear_idx,
        output_pdf=output_pdf,
        show=show,
    )

    

if __name__ == "__main__":
    regularisation_weight = 1e-0
    regularisation_weight_incl_colouration = 1e-0
    run_dir_loc_init_2_r_2, _, _, title_loc, analysis_loc = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e2, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=regularisation_weight),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=regularisation_weight),
        },
        run_label="loc",
        plot_row_title="Localisation loss alone",
    )

    run_dir_unc_init_2_r_2, _, _, title_unc, analysis_unc = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "uncertainty": ListenerBasedLoss(model=unc_model, weight=1e-1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=regularisation_weight),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=regularisation_weight),
        },
        run_label="unc",
        plot_row_title="Uncertainty loss alone",
    )

    run_dir_col_init_2_r_2, _, _, title_col, analysis_col = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "colouration": ListenerBasedLoss(model=col_model, weight=1e-2, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=regularisation_weight_incl_colouration),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=regularisation_weight_incl_colouration),
        },
        run_label="col",
        plot_row_title="Colouration loss alone",
    )

    run_dir_loc_unc_col_init_2_r_2, lstn_pos, spk_pos, title_all, analysis_all = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=2e2, device=device, dtype=dtype),
            "uncertainty": ListenerBasedLoss(model=unc_model, weight=1e-2, device=device, dtype=dtype),
            "colouration": ListenerBasedLoss(model=col_model, weight=5e-2, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=regularisation_weight_incl_colouration),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=regularisation_weight_incl_colouration),
        },
        run_label="loc_unc_col",
        plot_row_title="Combined losses",
    )

    folders = [run_dir_loc_init_2_r_2,
               run_dir_unc_init_2_r_2,
               run_dir_col_init_2_r_2,
               run_dir_loc_unc_col_init_2_r_2]

    print_cross_run_analysis([analysis_loc, analysis_unc, analysis_col, analysis_all])

    row_titles = [title_loc, title_unc, title_col, title_all]
    lstn_pos_arr = np.array(lstn_pos)
    spk_pos_arr = np.array(spk_pos)
    DAFx2026_fig5_plots(folders, lstn_pos_arr, spk_pos_arr, row_titles=row_titles, ear_idx=0, output_pdf="DAFx2026_fig5_left_ear.pdf")
    DAFx2026_fig5_plots(folders, lstn_pos_arr, spk_pos_arr, row_titles=row_titles, ear_idx=1, output_pdf="DAFx2026_fig5_right_ear.pdf", show=False)




    
