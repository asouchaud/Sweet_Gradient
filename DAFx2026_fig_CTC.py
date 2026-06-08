# Enable MPS fallback for PyTorch on macOS
import os
import sofar as sf
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import interface libraries
from pathlib import Path
import json
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
from scripts.perceptual_models.ipd.ipd import interaural_phase_difference_matching
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
from scripts.saving.save_results import saver
from scripts.helpers.dsp import compute_reference_bin




# ---------------------------
# Inversion
# ---------------------------
def compute_xtc_filters(H, reg=0):
    N = H.shape[-1]
    G = np.zeros_like(H, dtype=np.complex64)
    
    for k in range(N):
        Hf = H[:, :, k]
        G[:, :, k] = np.linalg.inv(Hf + reg * np.eye(2))
    
    return G


def sofa_find(source_angle, sofaile):
    pos = np.mod(np.array(sofaile.SourcePosition[:, :2]),360)
       
    source_angle = np.array([[source_angle%360, 0]])
    diff = source_angle[:, None, :] - pos[None, :, :]
    dists = np.linalg.vector_norm(diff, axis=2)
    # dists = np.minimum(dists, 360-dists)  # account for circularity
    return np.argmin(dists, axis=1)

def Compute_CTC_filters(target_angle):

    N = 256

    _sofa_symmetric = (
        Path(__file__).resolve().parent
        / "data"
        / "impulses"
        / "SOFA"
        / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
    )
    sofaile = sf.read_sofa(str(_sofa_symmetric))
    left_spk_hrirs = sofaile.Data_IR[760] # 30 deg
    right_spk_hrirs = sofaile.Data_IR[8064] # 330 deg
    ref_idx = sofa_find(target_angle, sofaile)
    ref_hrirs = np.squeeze(sofaile.Data_IR[ref_idx]) # 0 deg 
    ref_hrtf = np.fft.fft(ref_hrirs, n=N, axis = -1)

    IRs = np.concatenate((left_spk_hrirs[:,None,:], right_spk_hrirs[:,None,:]), axis = 1)
    H = np.fft.fft(IRs, n=N, axis = -1)
    # H, IRs = build_transfer_matrix()
    G = compute_xtc_filters(H)

    L_f = G[0,0] * ref_hrtf[0] + G[0,1] * ref_hrtf[1]
    R_f = G[1,0] * ref_hrtf[0] + G[1,1] * ref_hrtf[1]

    L = np.real(np.fft.ifft(L_f, axis = -1)).reshape(1,-1)
    R = np.real(np.fft.ifft(R_f, axis = -1)).reshape(1,-1)

    filts = np.concatenate((L,R), axis = 0)
    filts = np.fft.fftshift(filts, axes=-1)

    return(filts)

def remove_delay(spectrum, freqs_hz, delay_samples, fs=48000):
    """Advance ``spectrum`` (rfft) by ``delay_samples`` to align it to a reference."""
    return spectrum * np.exp(1j * 2 * np.pi * freqs_hz * delay_samples / fs)


def wrap_to_pi(x):
    """Match ``interaural_phase_difference_matching._wrap_to_pi``."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def wrapped_phase_deg(spectrum):
    """Wrapped phase in degrees from a complex RFFT spectrum."""
    return np.rad2deg(wrap_to_pi(np.angle(spectrum)))


def unwrapped_phase_deg(spectrum):
    """Unwrapped phase in degrees from a complex RFFT spectrum (along frequency)."""
    return np.rad2deg(np.unwrap(np.angle(spectrum), axis=-1))


def compute_ipd_spectrum(bin_lr, ref_lr, fs, n_fft=4096):
    """
    IPD spectra using the same definition as ``ipd.py``:
    IPD(f) = wrap( angle(L) - angle(R) ), error = wrap(IPD_bin - IPD_ref).

    Also returns unwrapped traces (unwrap on angle(L) - angle(R) before wrapping)
    for plotting cumulative interaural phase vs frequency.

    Inputs are (2, T) numpy arrays (left, right).
    """
    bin_f = np.fft.rfft(bin_lr, n=n_fft, axis=-1)
    ref_f = np.fft.rfft(ref_lr, n=n_fft, axis=-1)
    ipd_bin_raw = np.angle(bin_f[0]) - np.angle(bin_f[1])
    ipd_ref_raw = np.angle(ref_f[0]) - np.angle(ref_f[1])

    ipd_bin = wrap_to_pi(ipd_bin_raw)
    ipd_ref = wrap_to_pi(ipd_ref_raw)
    ipd_err = wrap_to_pi(ipd_bin - ipd_ref)

    ipd_bin_uw = np.unwrap(ipd_bin_raw)
    ipd_ref_uw = np.unwrap(ipd_ref_raw)
    ipd_err_uw = ipd_bin_uw - ipd_ref_uw

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return ipd_bin, ipd_ref, ipd_err, freqs, ipd_bin_uw, ipd_ref_uw, ipd_err_uw


def mean_abs_ipd_error_deg(ipd_err, freqs, fmin=200.0, fmax=7000.0):
    """Mean absolute wrapped IPD error in degrees (IPD loss band)."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.rad2deg(np.mean(np.abs(ipd_err[mask]))))


def rms_normalize(x):
    """Unit-RMS scaling for fair spectral-shape comparison (CTC vs proposed vs reference)."""
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x**2))
    if rms <= 0:
        return x
    return x / rms


def log_magnitude_db(x, n_fft, fs, eps=1e-8):
    """Log-magnitude spectrum in dB, matching ``spectral_matching._log_spectral_distortion``."""
    spec = np.fft.rfft(np.asarray(x), n=int(n_fft), axis=-1)
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(fs))
    mag_db = 20.0 * np.log10(np.abs(spec) + float(eps))
    return mag_db, freqs


def colouration_model_score(col_model, bin_lr, reference_stim, listener_indices):
    """Scalar colouration (LSD) from the trained perceptual module."""
    with torch.no_grad():
        bin_stim = (
            torch.as_tensor(bin_lr, device=col_model.device, dtype=col_model.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return float(col_model(bin_stim, reference_stim, listener_indices).mean().item())


def ipd_model_score(ipd_model, bin_stim, reference_stim, listener_indices):
    """Scalar IPD loss from the trained perceptual module (radians, mean |error|)."""
    with torch.no_grad():
        return float(ipd_model(bin_stim, reference_stim, listener_indices).mean().item())


def smooth_third_octave(freq, mag_db):
    smoothed = np.zeros_like(mag_db)

    k = 2**(1/6)  # 1/3 octave half-width

    for i, fc in enumerate(freq):
        f_low = fc / k
        f_high = fc * k

        idx = np.where((freq >= f_low) & (freq <= f_high))[0]

        smoothed[i] = np.mean(mag_db[idx])

    return smoothed


def _shared_ylim(axes, pad_frac=0.06):
    """Common y-limits for a group of axes from all lines within each x-range."""
    ys = []
    for ax in axes:
        xlo, xhi = ax.get_xlim()
        for line in ax.get_lines():
            x, y = line.get_data()
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            mask = (x >= xlo) & (x <= xhi) & np.isfinite(y)
            if np.any(mask):
                ys.append(y[mask])
    if not ys:
        return
    vals = np.concatenate(ys)
    y0 = float(np.min(vals))
    y1 = float(np.max(vals))
    if y1 <= y0:
        y1 = y0 + 1.0
    pad = (y1 - y0) * pad_frac
    ylim = (y0 - pad, y1 + pad)
    for ax in axes:
        ax.set_ylim(ylim)


def _autoscale_y(ax, pad_frac=0.06):
    """Per-axis y-limits (e.g. IPD panels)."""
    _shared_ylim([ax], pad_frac=pad_frac)


def _resolve_device(device_name):
    if isinstance(device_name, torch.device):
        return device_name
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _load_run_config(run_dir):
    with open(Path(run_dir) / "config.json", encoding="utf-8") as f:
        return json.load(f)


def build_plot_objects_from_config(main_cfg, repo_root=None):
    """Rebuild propagation and perceptual models for fig. 7 from a saved config."""
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    device = _resolve_device(main_cfg.get("device", "cpu"))
    dtype = getattr(torch, str(main_cfg.get("dtype", "float32")).split(".")[-1])

    lstn_pos = torch.tensor(main_cfg["lstn_pos"], device=device, dtype=dtype)
    spk_pos = torch.tensor(main_cfg["spk_pos"], device=device, dtype=dtype)
    sofa_path = main_cfg.get(
        "sofa_path",
        str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
    )

    prop = BinauralFreeFieldPropagation(
        fs=main_cfg["fs"],
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        sofa_file_path=sofa_path,
        device=device,
        dtype=dtype,
    )
    trainer = SimpleNamespace(prop=prop, device=device, dtype=dtype)

    col_model = spectral_matching(
        loss=F.l1_loss,
        target=0,
        target_pos=main_cfg["target_pos"],
        fs=main_cfg["fs"],
        sofa_path=sofa_path,
        lstn_pos=lstn_pos,
        erb_weighting=True,
        normalize_log_spectrum=False,
        fmin=0.0,
        fmax=24000.0,
        device=device,
        dtype=dtype,
    )
    ipd_model = interaural_phase_difference_matching(
        loss=F.l1_loss,
        target=0,
        target_pos=main_cfg["target_pos"],
        fs=main_cfg["fs"],
        sofa_path=sofa_path,
        lstn_pos=lstn_pos,
        fmin=0.0,
        fmax=24000.0,
        erb_weighting=main_cfg.get("ipd_erb_weighting", False),
        use_mse=main_cfg.get("ipd_use_mse", True),
        device=device,
        dtype=dtype,
    )
    return trainer, col_model, ipd_model


def regenerate_fig7_from_run(run_dir, run_dir_col_only, output_dir=None, show=False):
    """Recreate fig. 7 PDFs from saved artifacts and config.json (no retraining)."""
    run_dir = Path(run_dir)
    run_dir_col_only = Path(run_dir_col_only)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_cfg = _load_run_config(run_dir).get("main", {})
    trainer, col_model, ipd_model = build_plot_objects_from_config(main_cfg)

    ctc_path = run_dir / "artifacts" / "true_ctc_filters.npy"
    CTC_filts = np.load(ctc_path) if ctc_path.is_file() else Compute_CTC_filters(
        target_angle=main_cfg["target_angle"]
    )

    fig7_plot, fig7_phase_plot, fig7_ipd_plot = DAFx2026_fig7_plots(
        run_dir,
        run_dir_col_only,
        CTC_filts=CTC_filts,
        trainer=trainer,
        col_model=col_model,
        ipd_model=ipd_model,
    )

    outputs = {
        "DAFx2026_fig7.pdf": fig7_plot,
        "DAFx2026_fig7_phase.pdf": fig7_phase_plot,
        "DAFx2026_fig7_ipd.pdf": fig7_ipd_plot,
    }
    saved = []
    for name, fig in outputs.items():
        out_path = output_dir / name
        fig.savefig(out_path, format="pdf")
        saved.append(out_path)
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return saved


def DAFx2026_fig7_plots(run_dir, run_dir_col_only, CTC_filts, trainer, col_model, ipd_model=None):

    folder_name = run_dir
    filt_path = os.path.join(folder_name, "artifacts", "est_filters.npy")
    filts = np.load(filt_path)
    N_filts = filts.shape[-1]

    filt_path_col_only = os.path.join(run_dir_col_only, "artifacts", "est_filters.npy")
    filts_col_only = np.load(filt_path_col_only)

    # Metrics use raw ear signals (training). Magnitude plots RMS-normalise every
    # curve the same way so CTC, proposed, and reference are comparable in level.
    fs_plot = int(trainer.prop.fs)
    n_fft_col = int(col_model.n_fft)
    eps_col = float(col_model.eps)
    col_band = (float(col_model.fmin), float(col_model.fmax))
    col_mask = col_model.freq_bin_mask.detach().cpu().numpy()

    ctc_nobp_path = os.path.join(folder_name, "artifacts", "true_ctc_filters.npy")
    if os.path.isfile(ctc_nobp_path):
        ctc_raw = np.load(ctc_nobp_path)
    else:
        ctc_raw = np.asarray(CTC_filts)

    def _to_filt_torch(f):
        return torch.as_tensor(f, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)

    plot_prop = BinauralFreeFieldPropagation(
        fs=trainer.prop.fs,
        lstn_pos=col_model.lstn_pos,
        spk_pos=trainer.prop.spk_pos,
        sofa_file_path=trainer.prop.sources[0].sofa_file_path,
        apply_distance_attenuation=trainer.prop.apply_distance_attenuation,
        device=trainer.device,
        dtype=trainer.dtype,
    )
    if plot_prop.n_listeners != col_model.reference_hrirs.shape[0]:
        raise ValueError("Mismatch between plotting propagation listeners and reference_hrirs.")
    eval_listener_idx = torch.arange(col_model.reference_hrirs.shape[0], device=trainer.device)
    centre_listener_idx = int(len(eval_listener_idx) // 2)
    centre_li = torch.tensor([centre_listener_idx], device=trainer.device)

    reference_stim = torch.zeros(1, 1, int(0.01 * fs_plot), device=trainer.device, dtype=trainer.dtype)
    reference_stim[0, 0, 0] = 1.0

    def _propagate(filts_np):
        out = plot_prop(reference_stim, _to_filt_torch(filts_np), eval_listener_idx).cpu().numpy()[0]
        return out[centre_listener_idx]

    bin_stim_ctc = _propagate(ctc_raw)
    bin_stim_filts = _propagate(filts)
    bin_stim_filts_col_only = _propagate(filts_col_only)

    ref = compute_reference_bin(
        reference_stim=reference_stim,
        reference_hrirs=col_model.reference_hrirs,
        listener_indices=eval_listener_idx,
    ).cpu().numpy()[centre_listener_idx]

    # --- Magnitude plots: RMS-normalised (comparable shapes / overlay scale) ---
    ctc_f_plot = rms_normalize(ctc_raw)
    filts_f_plot = rms_normalize(filts)
    filts_col_only_f_plot = rms_normalize(filts_col_only)
    ref_plot = rms_normalize(ref)
    bin_stim_ctc_plot = rms_normalize(bin_stim_ctc)
    bin_stim_filts_plot = rms_normalize(bin_stim_filts)
    bin_stim_col_only_plot = rms_normalize(bin_stim_filts_col_only)

    CTC_f, freqs_filt = log_magnitude_db(ctc_f_plot, N_filts, fs_plot, eps_col)
    filts_f, _ = log_magnitude_db(filts_f_plot, N_filts, fs_plot, eps_col)
    filts_col_only_f, _ = log_magnitude_db(filts_col_only_f_plot, N_filts, fs_plot, eps_col)

    lsd_filt = np.sqrt(np.mean((CTC_f - filts_f) ** 2, axis=-1)).mean()
    print(f"Filter log-magnitude LSD (CTC vs proposed, RMS-normalised): {lsd_filt:.2f} dB")

    CTC_f_smooth_left = smooth_third_octave(freqs_filt, CTC_f[0])
    CTC_f_smooth_right = smooth_third_octave(freqs_filt, CTC_f[1])
    filts_f_smooth_left = smooth_third_octave(freqs_filt, filts_f[0])
    filts_f_smooth_right = smooth_third_octave(freqs_filt, filts_f[1])
    filts_col_only_f_smooth_left = smooth_third_octave(freqs_filt, filts_col_only_f[0])
    filts_col_only_f_smooth_right = smooth_third_octave(freqs_filt, filts_col_only_f[1])

    ref_f, freqs_ear = log_magnitude_db(ref_plot, n_fft_col, fs_plot, eps_col)
    bin_stim_filts_f, _ = log_magnitude_db(bin_stim_filts_plot, n_fft_col, fs_plot, eps_col)
    bin_stim_filts_col_only_f, _ = log_magnitude_db(bin_stim_col_only_plot, n_fft_col, fs_plot, eps_col)
    bin_stim_ctc_f, _ = log_magnitude_db(bin_stim_ctc_plot, n_fft_col, fs_plot, eps_col)

    freqs_plot = freqs_ear[col_mask]
    ref_f_smooth_left = smooth_third_octave(freqs_plot, ref_f[0, col_mask])
    ref_f_smooth_right = smooth_third_octave(freqs_plot, ref_f[1, col_mask])
    bin_stim_ctc_f_smooth_left = smooth_third_octave(freqs_plot, bin_stim_ctc_f[0, col_mask])
    bin_stim_ctc_f_smooth_right = smooth_third_octave(freqs_plot, bin_stim_ctc_f[1, col_mask])
    bin_stim_filts_f_smooth_left = smooth_third_octave(freqs_plot, bin_stim_filts_f[0, col_mask])
    bin_stim_filts_f_smooth_right = smooth_third_octave(freqs_plot, bin_stim_filts_f[1, col_mask])
    bin_stim_filts_col_only_f_smooth_left = smooth_third_octave(
        freqs_plot, bin_stim_filts_col_only_f[0, col_mask]
    )
    bin_stim_filts_col_only_f_smooth_right = smooth_third_octave(
        freqs_plot, bin_stim_filts_col_only_f[1, col_mask]
    )

    s_filts = colouration_model_score(col_model, bin_stim_filts, reference_stim, centre_li)
    s_col = colouration_model_score(col_model, bin_stim_filts_col_only, reference_stim, centre_li)
    s_ctc = colouration_model_score(col_model, bin_stim_ctc, reference_stim, centre_li)
    print(
        f"Colouration LSD (raw signals, training metric, [{col_band[0]:.0f},{col_band[1]:.0f}] Hz) -- "
        f"Col.+IPD: {s_filts:.2f} dB, Col. only: {s_col:.2f} dB, CTC: {s_ctc:.2f} dB"
    )

    # Phase plots: unwrapped phase (degrees). Ear spectra: remove bulk filter delay first.
    ref_fft = np.fft.rfft(ref, n=n_fft_col, axis=-1)
    bin_stim_filts_fft = np.fft.rfft(bin_stim_filts, n=n_fft_col, axis=-1)
    bin_stim_filts_col_only_fft = np.fft.rfft(bin_stim_filts_col_only, n=n_fft_col, axis=-1)
    bin_stim_ctc_fft = np.fft.rfft(bin_stim_ctc, n=n_fft_col, axis=-1)
    filts_fft = np.fft.rfft(filts_f_plot, n=N_filts, axis=-1)
    filts_col_only_fft = np.fft.rfft(filts_col_only_f_plot, n=N_filts, axis=-1)
    CTC_fft = np.fft.rfft(ctc_f_plot, n=N_filts, axis=-1)

    CTC_p = unwrapped_phase_deg(CTC_fft)
    filts_p = unwrapped_phase_deg(filts_fft)
    filts_col_only_p = unwrapped_phase_deg(filts_col_only_fft)

    CTC_p_smooth_left = smooth_third_octave(freqs_filt, CTC_p[0])
    CTC_p_smooth_right = smooth_third_octave(freqs_filt, CTC_p[1])
    filts_p_smooth_left = smooth_third_octave(freqs_filt, filts_p[0])
    filts_p_smooth_right = smooth_third_octave(freqs_filt, filts_p[1])
    filts_col_only_p_smooth_left = smooth_third_octave(freqs_filt, filts_col_only_p[0])
    filts_col_only_p_smooth_right = smooth_third_octave(freqs_filt, filts_col_only_p[1])

    filter_delay_samples = N_filts // 2
    bin_stim_filts_fft_al = remove_delay(bin_stim_filts_fft, freqs_ear, filter_delay_samples)
    bin_stim_filts_col_only_fft_al = remove_delay(
        bin_stim_filts_col_only_fft, freqs_ear, filter_delay_samples
    )
    bin_stim_ctc_fft_al = remove_delay(bin_stim_ctc_fft, freqs_ear, filter_delay_samples)

    ref_p = unwrapped_phase_deg(ref_fft)
    bin_stim_filts_p = unwrapped_phase_deg(bin_stim_filts_fft_al)
    bin_stim_filts_col_only_p = unwrapped_phase_deg(bin_stim_filts_col_only_fft_al)
    bin_stim_ctc_p = unwrapped_phase_deg(bin_stim_ctc_fft_al)

    ref_p_smooth_left = smooth_third_octave(freqs_plot, ref_p[0, col_mask])
    ref_p_smooth_right = smooth_third_octave(freqs_plot, ref_p[1, col_mask])
    bin_stim_ctc_p_smooth_left = smooth_third_octave(freqs_plot, bin_stim_ctc_p[0, col_mask])
    bin_stim_ctc_p_smooth_right = smooth_third_octave(freqs_plot, bin_stim_ctc_p[1, col_mask])
    bin_stim_filts_p_smooth_left = smooth_third_octave(freqs_plot, bin_stim_filts_p[0, col_mask])
    bin_stim_filts_p_smooth_right = smooth_third_octave(freqs_plot, bin_stim_filts_p[1, col_mask])
    bin_stim_filts_col_only_p_smooth_left = smooth_third_octave(
        freqs_plot, bin_stim_filts_col_only_p[0, col_mask]
    )
    bin_stim_filts_col_only_p_smooth_right = smooth_third_octave(
        freqs_plot, bin_stim_filts_col_only_p[1, col_mask]
    )

    filt_xlim = (80.0, 20000.0)
    ear_xlim = (max(80.0, col_band[0]), min(20000.0, col_band[1]))

    fontsize = 16
    fontsize_title = 13
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Magnitude (RMS-normalised)", fontsize=fontsize_title, y=0.98)

    ax1 = fig.add_subplot(2, 2, 1)
    l1, = ax1.semilogx(freqs_filt, filts_f_smooth_left, label="Proposed Method", linewidth=3)
    l1b, = ax1.semilogx(freqs_filt, filts_col_only_f_smooth_left, label="Proposed Method (Col. only)", linewidth=3)
    l2, = ax1.semilogx(freqs_filt, CTC_f_smooth_left, label="CTC", linewidth=3)
    ax1.grid()
    ax1.set_ylabel("Magnitude [dB]", fontsize=fontsize)
    ax1.set_title("Left filter", fontsize=fontsize_title)
    ax1.set_xlim(filt_xlim)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Right filter", fontsize=fontsize_title)
    ax2.semilogx(freqs_filt, filts_f_smooth_right, linewidth=3)
    ax2.semilogx(freqs_filt, filts_col_only_f_smooth_right, linewidth=3)
    ax2.semilogx(freqs_filt, CTC_f_smooth_right, linewidth=3)
    ax2.grid()
    ax2.set_xlim(filt_xlim)
    _shared_ylim([ax1, ax2])

    ax3 = fig.add_subplot(2, 2, 3)
    l3, = ax3.semilogx(freqs_plot, bin_stim_filts_f_smooth_left, linewidth=3)
    l3b, = ax3.semilogx(freqs_plot, bin_stim_filts_col_only_f_smooth_left, linewidth=3)
    l4, = ax3.semilogx(freqs_plot, bin_stim_ctc_f_smooth_left, linewidth=3)
    l5, = ax3.semilogx(freqs_plot, ref_f_smooth_left, label="Reference", linewidth=4, color="black", linestyle="--")
    ax3.grid()
    ax3.set_xlim(ear_xlim)
    ax3.set_title("Left ear", fontsize=fontsize_title)
    ax3.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax3.set_ylabel("Magnitude [dB]", fontsize=fontsize)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.semilogx(freqs_plot, bin_stim_filts_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_plot, bin_stim_filts_col_only_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_plot, bin_stim_ctc_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_plot, ref_f_smooth_right, linewidth=4, color="black", linestyle="--")
    ax4.grid()
    ax4.set_xlim(ear_xlim)
    ax4.set_title("Right ear", fontsize=fontsize_title)
    ax4.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    _shared_ylim([ax3, ax4])

    fig.legend(
        [l1, l1b, l2, l5],
        ["Col. + IPD", "Col. only", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        fontsize=fontsize_title,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # ---------------------------
    # Phase figure (mirrors the magnitude figure layout)
    # ---------------------------
    fig_phase = plt.figure(figsize=(8, 6))
    fig_phase.suptitle("Phase (unwrapped)", fontsize=fontsize_title, y=0.98)

    axp1 = fig_phase.add_subplot(2, 2, 1)
    lp1, = axp1.semilogx(freqs_filt, filts_p_smooth_left, label="Proposed Method", linewidth=3)
    lp1b, = axp1.semilogx(
        freqs_filt, filts_col_only_p_smooth_left, label="Proposed Method (Col. only)", linewidth=3
    )
    lp2, = axp1.semilogx(freqs_filt, CTC_p_smooth_left, label="CTC", linewidth=3)
    axp1.grid()
    axp1.set_ylabel("Phase [deg]", fontsize=fontsize)
    axp1.set_title("Left filter", fontsize=fontsize_title)
    axp1.set_xlim(filt_xlim)

    axp2 = fig_phase.add_subplot(2, 2, 2)
    axp2.set_title("Right filter", fontsize=fontsize_title)
    axp2.semilogx(freqs_filt, filts_p_smooth_right, linewidth=3)
    axp2.semilogx(freqs_filt, filts_col_only_p_smooth_right, linewidth=3)
    axp2.semilogx(freqs_filt, CTC_p_smooth_right, linewidth=3)
    axp2.grid()
    axp2.set_xlim(filt_xlim)
    _shared_ylim([axp1, axp2])

    axp3 = fig_phase.add_subplot(2, 2, 3)
    axp3.semilogx(freqs_plot, bin_stim_filts_p_smooth_left, linewidth=3)
    axp3.semilogx(freqs_plot, bin_stim_filts_col_only_p_smooth_left, linewidth=3)
    axp3.semilogx(freqs_plot, bin_stim_ctc_p_smooth_left, linewidth=3)
    lp5, = axp3.semilogx(
        freqs_plot, ref_p_smooth_left, label="Reference", linewidth=4, color="black", linestyle="--"
    )
    axp3.grid()
    axp3.set_xlim(ear_xlim)
    axp3.set_title("Left ear", fontsize=fontsize_title)
    axp3.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    axp3.set_ylabel("Phase [deg]", fontsize=fontsize)

    axp4 = fig_phase.add_subplot(2, 2, 4)
    axp4.semilogx(freqs_plot, bin_stim_filts_p_smooth_right, linewidth=3)
    axp4.semilogx(freqs_plot, bin_stim_filts_col_only_p_smooth_right, linewidth=3)
    axp4.semilogx(freqs_plot, bin_stim_ctc_p_smooth_right, linewidth=3)
    axp4.semilogx(
        freqs_plot, ref_p_smooth_right, linewidth=4, color="black", linestyle="--"
    )
    axp4.grid()
    axp4.set_xlim(ear_xlim)
    axp4.set_title("Right ear", fontsize=fontsize_title)
    axp4.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    _shared_ylim([axp3, axp4])

    fig_phase.legend(
        [lp1, lp1b, lp2, lp5],
        ["Col. + IPD", "Col. only", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        fontsize=fontsize_title,
        frameon=False,
    )

    fig_phase.tight_layout(rect=[0, 0, 1, 0.90])

    # ---------------------------
    # Interaural Phase Difference (IPD) figure
    # ---------------------------
    if ipd_model is not None:
        n_fft_ipd = int(ipd_model.n_fft)
        ipd_band = (float(ipd_model.fmin), float(ipd_model.fmax))
    else:
        n_fft_ipd = 4096
        ipd_band = (50.0, 15000.0)
    fontsize_ipd = 13

    _, _, _, freqs_ipd, ipd_ref_uw, _, _ = compute_ipd_spectrum(ref, ref, fs_plot, n_fft_ipd)
    _, _, err_ctc, _, ipd_ctc_uw, _, err_ctc_uw = compute_ipd_spectrum(
        bin_stim_ctc, ref, fs_plot, n_fft_ipd
    )
    _, _, err_filts, _, ipd_filts_uw, _, err_filts_uw = compute_ipd_spectrum(
        bin_stim_filts, ref, fs_plot, n_fft_ipd
    )
    _, _, err_col_only, _, ipd_col_only_uw, _, err_col_only_uw = compute_ipd_spectrum(
        bin_stim_filts_col_only, ref, fs_plot, n_fft_ipd
    )

    ipd_ref_deg = np.rad2deg(ipd_ref_uw)
    ipd_ctc_deg = np.rad2deg(ipd_ctc_uw)
    ipd_filts_deg = np.rad2deg(ipd_filts_uw)
    ipd_col_only_deg = np.rad2deg(ipd_col_only_uw)
    err_ctc_deg = np.rad2deg(err_ctc_uw)
    err_filts_deg = np.rad2deg(err_filts_uw)
    err_col_only_deg = np.rad2deg(err_col_only_uw)

    m_filts = mean_abs_ipd_error_deg(err_filts, freqs_ipd, *ipd_band)
    m_col = mean_abs_ipd_error_deg(err_col_only, freqs_ipd, *ipd_band)
    m_ctc = mean_abs_ipd_error_deg(err_ctc, freqs_ipd, *ipd_band)
    print(
        f"Mean |IPD error| in [{ipd_band[0]:.0f},{ipd_band[1]:.0f}] Hz (raw signals, training band) -- "
        f"Col.+IPD: {m_filts:.1f} deg, Col. only: {m_col:.1f} deg, CTC: {m_ctc:.1f} deg"
    )

    if ipd_model is not None:
        def _bin_torch(x):
            return torch.tensor(x, device=trainer.device, dtype=trainer.dtype).unsqueeze(0).unsqueeze(0)

        s_mod = ipd_model_score(ipd_model, _bin_torch(bin_stim_filts), reference_stim, centre_li)
        s_col = ipd_model_score(ipd_model, _bin_torch(bin_stim_filts_col_only), reference_stim, centre_li)
        s_ctc = ipd_model_score(ipd_model, _bin_torch(bin_stim_ctc), reference_stim, centre_li)
        print(
            f"IPD module (rad, mean |err|) -- Col.+IPD: {s_mod:.4f}, Col. only: {s_col:.4f}, "
            f"CTC: {s_ctc:.4f}  (as deg: {np.rad2deg(s_mod):.1f}, {np.rad2deg(s_col):.1f}, {np.rad2deg(s_ctc):.1f})"
        )

    fig_ipd = plt.figure(figsize=(9, 4))
    f_plot_lo = max(80.0, ipd_band[0] * 0.5)
    f_plot_hi = min(20000.0, ipd_band[1] * 1.2)
    axi1 = fig_ipd.add_subplot(1, 2, 1)
    li1, = axi1.semilogx(freqs_ipd, ipd_filts_deg, linewidth=2.5)
    li1b, = axi1.semilogx(freqs_ipd, ipd_col_only_deg, linewidth=2.5)
    li2, = axi1.semilogx(freqs_ipd, ipd_ctc_deg, linewidth=2.5)
    li5, = axi1.semilogx(freqs_ipd, ipd_ref_deg, linewidth=3, color="black", linestyle="--")
    axi1.grid()
    axi1.set_xlim([f_plot_lo, f_plot_hi])
    axi1.set_title("IPD (unwrapped)", fontsize=fontsize_ipd)
    _autoscale_y(axi1)
    axi1.set_xlabel("Frequency [Hz]", fontsize=fontsize_ipd)
    axi1.set_ylabel("Unwrapped IPD [deg]", fontsize=fontsize_ipd)

    axi2 = fig_ipd.add_subplot(1, 2, 2)
    axi2.semilogx(freqs_ipd, err_filts_deg, linewidth=2.5)
    axi2.semilogx(freqs_ipd, err_col_only_deg, linewidth=2.5)
    axi2.semilogx(freqs_ipd, err_ctc_deg, linewidth=2.5)
    axi2.axhline(0.0, color="black", linestyle="--", linewidth=2)
    axi2.grid()
    axi2.set_xlim([f_plot_lo, f_plot_hi])
    axi2.set_title("IPD error (unwrapped)", fontsize=fontsize_ipd)
    _autoscale_y(axi2)
    axi2.set_xlabel("Frequency [Hz]", fontsize=fontsize_ipd)
    axi2.set_ylabel("Unwrapped error [deg]", fontsize=fontsize_ipd)

    fig_ipd.legend(
        [li1, li1b, li2, li5],
        ["Col. + IPD", "Col. only", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=fontsize_ipd,
        frameon=False,
    )

    fig_ipd.tight_layout(rect=[0, 0, 1, 0.88])

    return fig, fig_phase, fig_ipd


def init_delta_fir(nn_model, n_samples):
    """Centre-tap delta initialisation for each source FIR branch."""
    centre_tap_idx = n_samples // 2
    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[centre_tap_idx, 0] = 1.0


def init_noise_fir(nn_model):
    """Gaussian noise initialisation for each source FIR branch."""
    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.copy_(torch.randn_like(branch.weight))


def init_fir(nn_model, n_samples, mode="noise"):
    """Initialise FIR weights. ``mode`` is ``'noise'`` or ``'delta'``."""
    if mode == "noise":
        init_noise_fir(nn_model)
    elif mode == "delta":
        init_delta_fir(nn_model, n_samples)
    else:
        raise ValueError(f"filter_init must be 'noise' or 'delta'; got {mode!r}.")


if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parent

    env = SimpleNamespace(
        target_angle=45.0,
        target_distance=100,
        epochs=5000,
        listener_batch_size=1,
        symmetric_listener_sampling=True,
        n_samples=256,
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.0005,
        seed=46,
        log=1,
        fs=48000,
        filter_init="delta",  # "noise" or "delta"
        ipd_erb_weighting=False,
    )
    env.target_pos = [
        env.target_distance,
        float(np.tan(np.deg2rad(env.target_angle)) * env.target_distance),
    ]
    env.dtype = torch.float32
    env.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {env.device}")

    env.lstn_pos = [[0.0, 0.0]]
    base_angle_rad = np.deg2rad(30.0)
    env.speaker_distance = 2.0
    env.spk_pos = [
        [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
        [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
    ]
    env.n_sources = len(env.spk_pos)
    env.col_weight = 1.0e0
    env.ipd_weight = 5.0e0

    set_seed(seed=env.seed)

    training_stimulus_data_loader, _ = get_stim_loader(
        env.fs, 1, 1, stimulus_type=env.selected_stim, device=env.device, dtype=env.dtype
    )
    env.lstn_pos = torch.tensor(env.lstn_pos, device=env.device, dtype=env.dtype)
    env.spk_pos = torch.tensor(env.spk_pos, device=env.device, dtype=env.dtype)
    training_listener_data_loader, _ = get_listener_loader(
        env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling
    )

    stereo_prop = BinauralFreeFieldPropagation(
        fs=env.fs, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos,
        sofa_file_path=env.sofa_path, device=env.device, dtype=env.dtype,
    )

    col_model = spectral_matching(
        loss=F.l1_loss, target=0, target_pos=env.target_pos, fs=env.fs,
        sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, erb_weighting=True,
        normalize_log_spectrum=False, fmin = 0.0, fmax = 24000.0, device=env.device, dtype=env.dtype,
    )
    ipd_model = interaural_phase_difference_matching(
        loss=F.l1_loss, target=0, target_pos=env.target_pos, fs=env.fs,
        sofa_path=env.sofa_path, lstn_pos=env.lstn_pos,
        fmin=0.0, fmax=24000.0, erb_weighting=env.ipd_erb_weighting,
        use_mse = True, 
        device=env.device, dtype=env.dtype,
    )

    loss_dict = {
        "colouration": ListenerBasedLoss(model=col_model, weight=env.col_weight, device=env.device, dtype=env.dtype),
        "ipd": ListenerBasedLoss(model=ipd_model, weight=env.ipd_weight, device=env.device, dtype=env.dtype),
    }

    nn_model = MultiSourceFIR(n_samples=env.n_samples, n_sources=env.n_sources).to(device=env.device, dtype=env.dtype)
    init_fir(nn_model, env.n_samples, mode=env.filter_init)
    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

    saver_obj = saver()
    trainer = Train(
        n_epoch=env.epochs,
        stim_loader=training_stimulus_data_loader,
        lstn_loader=training_listener_data_loader,
        nn=nn_model,
        prop=stereo_prop,
        loss_dict=loss_dict,
        optimiser=nn_optimiser,
        logger=saver_obj if env.log else None,
    )
    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for col+IPD optimisation: {env.time_train_seconds:.2f} s")

    eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(
        env,
        eval_lstn_pos=torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5]], device=env.device, dtype=env.dtype),
    )
    eval_ipd_model = interaural_phase_difference_matching(
        loss=F.l1_loss, target=0, target_pos=env.target_pos, fs=env.fs,
        sofa_path=env.sofa_path, lstn_pos=eval_lstn_pos,
        erb_weighting=env.ipd_erb_weighting, device=env.device, dtype=env.dtype,
    )
    eval_perceptual_dict["ipd"] = ListenerBasedLoss(
        model=eval_ipd_model, weight=1, device=env.device, dtype=env.dtype,
    )

    est_filters, estimation_dict, bin_stim = trainer.evaluate(
        eval_lstn_pos=eval_lstn_pos,
        eval_stimulus_data_loader=eval_stimulus_data_loader,
        loss_dict=eval_perceptual_dict,
    )
    est_filters /= np.max(np.abs(est_filters)) / 0.9

    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    saver_obj.save_results(estimation_dict=estimation_dict, est_filters=est_filters, fs=env.fs, env=env)

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
    plt.close("all")

    set_seed(seed=env.seed)
    nn_model_col_only = MultiSourceFIR(n_samples=env.n_samples, n_sources=env.n_sources).to(device=env.device, dtype=env.dtype)
    init_fir(nn_model_col_only, env.n_samples, mode=env.filter_init)
    loss_dict_col_only = {
        "colouration": ListenerBasedLoss(model=col_model, weight=env.col_weight, device=env.device, dtype=env.dtype),
    }

    saver_obj_col_only = saver()
    trainer_col_only = Train(
        n_epoch=env.epochs,
        stim_loader=training_stimulus_data_loader,
        lstn_loader=training_listener_data_loader,
        nn=nn_model_col_only,
        prop=stereo_prop,
        loss_dict=loss_dict_col_only,
        optimiser=torch.optim.Adam(nn_model_col_only.parameters(), lr=env.lr),
        logger=saver_obj_col_only if env.log else None,
    )
    time_train_start = time.time()
    trainer_col_only.train()
    env.time_train_seconds_col_only = time.time() - time_train_start
    print(f"Time for colouration-only optimisation: {env.time_train_seconds_col_only:.2f} s")

    est_filters_col_only, estimation_dict_col_only, _ = trainer_col_only.evaluate(
        eval_lstn_pos=eval_lstn_pos,
        eval_stimulus_data_loader=eval_stimulus_data_loader,
        loss_dict=eval_perceptual_dict,
    )
    est_filters_col_only /= np.max(np.abs(est_filters_col_only)) / 0.9
    saver_obj_col_only.save_results(
        estimation_dict=estimation_dict_col_only,
        est_filters=est_filters_col_only,
        fs=env.fs,
        env=env,
    )

    CTC_filts = Compute_CTC_filters(target_angle=env.target_angle)
    ctc_artifacts_dir = saver_obj.run_dir / "artifacts"
    ctc_artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(ctc_artifacts_dir / "true_ctc_filters.npy", CTC_filts)

    fig7_plot, fig7_phase_plot, fig7_ipd_plot = DAFx2026_fig7_plots(
        saver_obj.run_dir,
        saver_obj_col_only.run_dir,
        CTC_filts=CTC_filts,
        trainer=trainer,
        col_model=col_model,
        ipd_model=ipd_model,
    )
    plot_folder = saver_obj.get_plot_folder()
    fig7_plot.savefig(plot_folder / "DAFx2026_fig7.pdf", format="pdf")
    fig7_phase_plot.savefig(plot_folder / "DAFx2026_fig7_phase.pdf", format="pdf")
    fig7_ipd_plot.savefig(plot_folder / "DAFx2026_fig7_ipd.pdf", format="pdf")
    print(f"Fig.7 plots saved to {plot_folder}")


