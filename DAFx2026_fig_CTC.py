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
from scipy import stats as scipy_stats

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
from scripts.helpers.filters import bandpass_fir
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


def compute_ipd_spectrum(bin_lr, ref_lr, fs, n_fft=4096):
    """
    IPD spectra using the same definition as ``ipd.py``:
    IPD(f) = wrap( angle(L) - angle(R) ), error = wrap(IPD_bin - IPD_ref).
    Inputs are (2, T) numpy arrays (left, right).
    """
    bin_f = np.fft.rfft(bin_lr, n=n_fft, axis=-1)
    ref_f = np.fft.rfft(ref_lr, n=n_fft, axis=-1)
    ipd_bin = wrap_to_pi(np.angle(bin_f[0]) - np.angle(bin_f[1]))
    ipd_ref = wrap_to_pi(np.angle(ref_f[0]) - np.angle(ref_f[1]))
    ipd_err = wrap_to_pi(ipd_bin - ipd_ref)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return ipd_bin, ipd_ref, ipd_err, freqs


def mean_abs_ipd_error_deg(ipd_err, freqs, fmin=200.0, fmax=7000.0):
    """Mean absolute wrapped IPD error in degrees (IPD loss band)."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.rad2deg(np.mean(np.abs(ipd_err[mask]))))


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



def DAFx2026_fig7_plots(run_dir, run_dir_col_only, CTC_filts, trainer, col_model, ipd_model=None):

    folder_name = run_dir
    filt_path = os.path.join(folder_name, "artifacts", "est_filters.npy")
    filts = np.load(filt_path)
    N_filts =  filts.shape[-1]

    folder_name_col_only = run_dir_col_only
    filt_path_col_only = os.path.join(folder_name_col_only, "artifacts", "est_filters.npy")
    filts_col_only = np.load(filt_path_col_only)

    CTC_filts = bandpass_fir(CTC_filts, fs=env.fs)
    filts = bandpass_fir(filts, fs=env.fs)
    filts_col_only = bandpass_fir(filts_col_only, fs=env.fs)

    ctc_normed = CTC_filts / np.sqrt(np.mean(CTC_filts**2))
    filts_normed = filts / np.sqrt(np.mean(filts**2))
    filts_col_only_normed = filts_col_only / np.sqrt(np.mean(filts_col_only**2))

    ctc_filts_torch = torch.tensor(ctc_normed, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    filts_torch = torch.tensor(filts_normed, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    filts_col_only_torch = torch.tensor(filts_col_only_normed, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    
    # trainer.evaluate(...) rebinds trainer.prop to evaluation listeners.
    # Build a local propagation object so plotting uses the same listener set
    # as col_model/reference_hrirs and avoids listener index mismatch.
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
    reference_stim = torch.zeros(1, 1, int(0.01 * env.fs), device=trainer.device, dtype=trainer.dtype)
    reference_stim[0, 0, 0] = 1.0
    bin_stim_ctc = plot_prop(reference_stim, ctc_filts_torch, eval_listener_idx).cpu().numpy()[0]
    bin_stim_filts = plot_prop(reference_stim, filts_torch, eval_listener_idx).cpu().numpy()[0]
    bin_stim_filts_col_only = plot_prop(reference_stim, filts_col_only_torch, eval_listener_idx).cpu().numpy()[0]
    centre_listener_idx = int(len(eval_listener_idx) // 2)
    bin_stim_ctc = bin_stim_ctc[centre_listener_idx]
    bin_stim_filts = bin_stim_filts[centre_listener_idx]
    bin_stim_filts_col_only = bin_stim_filts_col_only[centre_listener_idx]
    
    ref = compute_reference_bin(
        reference_stim=reference_stim,
        reference_hrirs=col_model.reference_hrirs,
        listener_indices=eval_listener_idx,
    ).cpu().numpy()[centre_listener_idx]
    N_ref = ref.shape[-1]

    bin_stim_ctc_normed = bin_stim_ctc / np.sqrt(np.mean(bin_stim_ctc**2))
    bin_stim_filts_normed = bin_stim_filts / np.sqrt(np.mean(bin_stim_filts**2))
    bin_stim_filts_col_only_normed = bin_stim_filts_col_only / np.sqrt(np.mean(bin_stim_filts_col_only**2))
    ref_normed = ref / np.sqrt(np.mean(ref**2))

    def _rfft_norm(x, n):
        return np.fft.rfft(x, n=n, axis=-1, norm="ortho")

    CTC_fft = _rfft_norm(ctc_normed, n=N_filts)
    filts_fft = _rfft_norm(filts_normed, n=N_filts)
    filts_col_only_fft = _rfft_norm(filts_col_only_normed, n=N_filts)

    # Magnitude
    CTC_f = 20*np.log10(np.abs(CTC_fft))
    filts_f = 20*np.log10(np.abs(filts_fft))
    filts_col_only_f = 20*np.log10(np.abs(filts_col_only_fft))

    lsd_freq = np.sqrt(np.mean((CTC_f - filts_f) ** 2, axis=-1))
    lsd = lsd_freq.mean(axis=-1)
    print(f"Log Spectral Distance between CTC and proposed method filters: {lsd:.2f} dB")

    freqs_filt = np.fft.rfftfreq(N_filts, d = 1/48000)

    CTC_f_smooth_left = smooth_third_octave(freqs_filt, CTC_f[0])
    CTC_f_smooth_right = smooth_third_octave(freqs_filt, CTC_f[1])
    filts_f_smooth_left = smooth_third_octave(freqs_filt, filts_f[0])
    filts_f_smooth_right = smooth_third_octave(freqs_filt, filts_f[1])
    filts_col_only_f_smooth_left = smooth_third_octave(freqs_filt, filts_col_only_f[0])
    filts_col_only_f_smooth_right = smooth_third_octave(freqs_filt, filts_col_only_f[1])


    ref_fft = _rfft_norm(ref_normed, n=N_ref)
    bin_stim_filts_fft = _rfft_norm(bin_stim_filts_normed, n=N_ref)
    bin_stim_filts_col_only_fft = _rfft_norm(bin_stim_filts_col_only_normed, n=N_ref)
    bin_stim_ctc_fft = _rfft_norm(bin_stim_ctc_normed, n=N_ref)
    
    ref_f = 20*np.log10(np.abs(ref_fft))
    bin_stim_filts_f = 20*np.log10(np.abs(bin_stim_filts_fft))
    bin_stim_filts_col_only_f = 20*np.log10(np.abs(bin_stim_filts_col_only_fft))
    bin_stim_ctc_f = 20*np.log10(np.abs(bin_stim_ctc_fft))

    freqs_ref = np.fft.rfftfreq(N_ref, d = 1/48000)

    ref_f_smooth_left = smooth_third_octave(freqs_ref, ref_f[0])
    ref_f_smooth_right = smooth_third_octave(freqs_ref, ref_f[1])
    bin_stim_ctc_f_smooth_left = smooth_third_octave(freqs_ref, bin_stim_ctc_f[0])
    bin_stim_ctc_f_smooth_right = smooth_third_octave(freqs_ref, bin_stim_ctc_f[1])
    bin_stim_filts_f_smooth_left = smooth_third_octave(freqs_ref, bin_stim_filts_f[0])
    bin_stim_filts_f_smooth_right = smooth_third_octave(freqs_ref, bin_stim_filts_f[1])
    bin_stim_filts_col_only_f_smooth_left = smooth_third_octave(freqs_ref, bin_stim_filts_col_only_f[0])
    bin_stim_filts_col_only_f_smooth_right = smooth_third_octave(freqs_ref, bin_stim_filts_col_only_f[1])

    # Phase (unwrapped, in degrees) using the same FFTs as the magnitude plots
    CTC_p = np.rad2deg(np.unwrap(np.angle(CTC_fft), axis=-1))
    filts_p = np.rad2deg(np.unwrap(np.angle(filts_fft), axis=-1))
    filts_col_only_p = np.rad2deg(np.unwrap(np.angle(filts_col_only_fft), axis=-1))

    CTC_p_smooth_left = smooth_third_octave(freqs_filt, CTC_p[0])
    CTC_p_smooth_right = smooth_third_octave(freqs_filt, CTC_p[1])
    filts_p_smooth_left = smooth_third_octave(freqs_filt, filts_p[0])
    filts_p_smooth_right = smooth_third_octave(freqs_filt, filts_p[1])
    filts_col_only_p_smooth_left = smooth_third_octave(freqs_filt, filts_col_only_p[0])
    filts_col_only_p_smooth_right = smooth_third_octave(freqs_filt, filts_col_only_p[1])

    # The loudspeaker filters carry a fixed bulk delay that is purely an
    # artifact of how they are represented: the FIRs are initialised as a delta
    # at the centre tap (n_samples//2) and the CTC filters are fftshifted by
    # N/2. This latency is identical for both ears, so removing it does NOT
    # affect the interaural phase difference (ITD/IPD). We deliberately use this
    # single, ear-independent delay rather than a per-ear estimate: a per-ear
    # estimate would also absorb the acoustic interaural delay and erase exactly
    # the interaural information the IPD term optimises. After removing only the
    # filtering delay, the phase curves reveal the reproduced spectral and
    # interaural phase rather than the bulk filter latency.
    filter_delay_samples = N_filts // 2
    bin_stim_filts_fft_al = remove_delay(bin_stim_filts_fft, freqs_ref, filter_delay_samples)
    bin_stim_filts_col_only_fft_al = remove_delay(bin_stim_filts_col_only_fft, freqs_ref, filter_delay_samples)
    bin_stim_ctc_fft_al = remove_delay(bin_stim_ctc_fft, freqs_ref, filter_delay_samples)

    ref_p = np.rad2deg(np.unwrap(np.angle(ref_fft), axis=-1))
    bin_stim_filts_p = np.rad2deg(np.unwrap(np.angle(bin_stim_filts_fft_al), axis=-1))
    bin_stim_filts_col_only_p = np.rad2deg(np.unwrap(np.angle(bin_stim_filts_col_only_fft_al), axis=-1))
    bin_stim_ctc_p = np.rad2deg(np.unwrap(np.angle(bin_stim_ctc_fft_al), axis=-1))

    ref_p_smooth_left = smooth_third_octave(freqs_ref, ref_p[0])
    ref_p_smooth_right = smooth_third_octave(freqs_ref, ref_p[1])
    bin_stim_ctc_p_smooth_left = smooth_third_octave(freqs_ref, bin_stim_ctc_p[0])
    bin_stim_ctc_p_smooth_right = smooth_third_octave(freqs_ref, bin_stim_ctc_p[1])
    bin_stim_filts_p_smooth_left = smooth_third_octave(freqs_ref, bin_stim_filts_p[0])
    bin_stim_filts_p_smooth_right = smooth_third_octave(freqs_ref, bin_stim_filts_p[1])
    bin_stim_filts_col_only_p_smooth_left = smooth_third_octave(freqs_ref, bin_stim_filts_col_only_p[0])
    bin_stim_filts_col_only_p_smooth_right = smooth_third_octave(freqs_ref, bin_stim_filts_col_only_p[1])


    fontsize = 18
    fig = plt.figure(figsize=(8,6))

    ax1 = fig.add_subplot(2,2,1)
    l1, = ax1.semilogx(freqs_filt, filts_f_smooth_left, label="Proposed Method", linewidth=3)
    l1b, = ax1.semilogx(freqs_filt, filts_col_only_f_smooth_left, label="Proposed Method (Col. only)", linewidth=3)
    l2, = ax1.semilogx(freqs_filt, CTC_f_smooth_left, label="CTC", linewidth=3)
    ax1.grid()
    ax1.set_ylabel("Magnitude [dB]", fontsize=fontsize)
    ax1.set_title("Left Loudspeaker Filter", fontsize=fontsize)
    ax1.set_xlim([80,20000])
    ax1.set_ylim([-30,10])

    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title("Right Loudspeaker Filter", fontsize=fontsize)
    ax2.semilogx(freqs_filt, filts_f_smooth_right, linewidth=3)
    ax2.semilogx(freqs_filt, filts_col_only_f_smooth_right, linewidth=3)
    ax2.semilogx(freqs_filt, CTC_f_smooth_right, linewidth=3)
    ax2.grid()
    ax2.set_xlim([80,20000])
    ax2.set_ylim([-30,10])

    ax3 = fig.add_subplot(2,2,3)
    l3, = ax3.semilogx(freqs_ref, bin_stim_filts_f_smooth_left, linewidth=3)
    l3b, = ax3.semilogx(freqs_ref, bin_stim_filts_col_only_f_smooth_left, linewidth=3)
    l4, = ax3.semilogx(freqs_ref, bin_stim_ctc_f_smooth_left, linewidth=3)
    l5, = ax3.semilogx(freqs_ref, ref_f_smooth_left, label="Reference", linewidth=4, color="black", linestyle="--")
    ax3.grid()
    ax3.set_xlim([80,20000])
    ax3.set_ylim([-30,10])
    ax3.set_title("Left Ear Signal", fontsize=fontsize)
    ax3.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax3.set_ylabel("Magnitude [dB]", fontsize=fontsize)

    ax4 = fig.add_subplot(2,2,4)
    ax4.semilogx(freqs_ref, bin_stim_filts_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_ref, bin_stim_filts_col_only_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_ref, bin_stim_ctc_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_ref, ref_f_smooth_right, linewidth=4, color="black", linestyle="--")
    ax4.grid()
    ax4.set_xlim([80,20000])
    ax4.set_ylim([-30,10])
    ax4.set_title("Right Ear Signal", fontsize=fontsize)
    ax4.set_xlabel("Frequency [Hz]", fontsize=fontsize)

    # ONE common legend
    fig.legend(
        [l1, l1b, l2, l5],
        ["Proposed Method (Col. + IPD)", "Proposed Method (Col. only)", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        fontsize=fontsize,
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.84])  # leave space for legend

    # ---------------------------
    # Phase figure (mirrors the magnitude figure layout)
    # ---------------------------
    fig_phase = plt.figure(figsize=(8,6))

    axp1 = fig_phase.add_subplot(2,2,1)
    lp1, = axp1.semilogx(freqs_filt, filts_p_smooth_left, label="Proposed Method", linewidth=3)
    lp1b, = axp1.semilogx(freqs_filt, filts_col_only_p_smooth_left, label="Proposed Method (Col. only)", linewidth=3)
    lp2, = axp1.semilogx(freqs_filt, CTC_p_smooth_left, label="CTC", linewidth=3)
    axp1.grid()
    axp1.set_ylabel("Phase [deg]", fontsize=fontsize)
    axp1.set_title("Left Loudspeaker Filter", fontsize=fontsize)
    axp1.set_xlim([80,20000])

    axp2 = fig_phase.add_subplot(2,2,2)
    axp2.set_title("Right Loudspeaker Filter", fontsize=fontsize)
    axp2.semilogx(freqs_filt, filts_p_smooth_right, linewidth=3)
    axp2.semilogx(freqs_filt, filts_col_only_p_smooth_right, linewidth=3)
    axp2.semilogx(freqs_filt, CTC_p_smooth_right, linewidth=3)
    axp2.grid()
    axp2.set_xlim([80,20000])

    axp3 = fig_phase.add_subplot(2,2,3)
    axp3.semilogx(freqs_ref, bin_stim_filts_p_smooth_left, linewidth=3)
    axp3.semilogx(freqs_ref, bin_stim_filts_col_only_p_smooth_left, linewidth=3)
    axp3.semilogx(freqs_ref, bin_stim_ctc_p_smooth_left, linewidth=3)
    lp5, = axp3.semilogx(freqs_ref, ref_p_smooth_left, label="Reference", linewidth=4, color="black", linestyle="--")
    axp3.grid()
    axp3.set_xlim([80,20000])
    axp3.set_title("Left Ear Signal", fontsize=fontsize)
    axp3.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    axp3.set_ylabel("Phase [deg]", fontsize=fontsize)

    axp4 = fig_phase.add_subplot(2,2,4)
    axp4.semilogx(freqs_ref, bin_stim_filts_p_smooth_right, linewidth=3)
    axp4.semilogx(freqs_ref, bin_stim_filts_col_only_p_smooth_right, linewidth=3)
    axp4.semilogx(freqs_ref, bin_stim_ctc_p_smooth_right, linewidth=3)
    axp4.semilogx(freqs_ref, ref_p_smooth_right, linewidth=4, color="black", linestyle="--")
    axp4.grid()
    axp4.set_xlim([80,20000])
    axp4.set_title("Right Ear Signal", fontsize=fontsize)
    axp4.set_xlabel("Frequency [Hz]", fontsize=fontsize)

    fig_phase.legend(
        [lp1, lp1b, lp2, lp5],
        ["Proposed Method (Col. + IPD)", "Proposed Method (Col. only)", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        fontsize=fontsize,
        frameon=False
    )

    fig_phase.tight_layout(rect=[0, 0, 1, 0.84])  # leave space for legend

    # ---------------------------
    # Interaural Phase Difference (IPD) figure
    # ---------------------------
    # Per-ear phase can match the reference while IPD still fails: colouration
    # optimises magnitude per ear, not wrap(angle(L)-angle(R)). Metrics use the
    # same definition as ipd.py (n_fft=4096, 200-7000 Hz), on un-normalised ear
    # signals and filters *without* bandpass so they match training/evaluation.
    n_fft_ipd = 4096
    ipd_band = (200.0, 7000.0)
    fontsize_ipd = 13
    fs_plot = int(trainer.prop.fs)

    filts_raw = np.load(filt_path)
    filts_col_only_raw = np.load(filt_path_col_only)
    ctc_nobp_path = os.path.join(folder_name, "artifacts", "true_ctc_filters.npy")
    if os.path.isfile(ctc_nobp_path):
        ctc_raw = np.load(ctc_nobp_path)
    else:
        ctc_raw = Compute_CTC_filters(target_angle=env.target_angle)
    filts_raw_t = torch.tensor(filts_raw, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    filts_col_only_raw_t = torch.tensor(filts_col_only_raw, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    ctc_raw_t = torch.tensor(ctc_raw, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    bin_stim_filts_raw = plot_prop(reference_stim, filts_raw_t, eval_listener_idx).cpu().numpy()[0][centre_listener_idx]
    bin_stim_col_only_raw = plot_prop(reference_stim, filts_col_only_raw_t, eval_listener_idx).cpu().numpy()[0][centre_listener_idx]
    bin_stim_ctc_raw = plot_prop(reference_stim, ctc_raw_t, eval_listener_idx).cpu().numpy()[0][centre_listener_idx]

    ipd_ref, _, _, freqs_ipd = compute_ipd_spectrum(ref, ref, fs_plot, n_fft_ipd)
    ipd_ctc, _, err_ctc, _ = compute_ipd_spectrum(bin_stim_ctc_raw, ref, fs_plot, n_fft_ipd)
    ipd_filts, _, err_filts, _ = compute_ipd_spectrum(bin_stim_filts_raw, ref, fs_plot, n_fft_ipd)
    ipd_col_only, _, err_col_only, _ = compute_ipd_spectrum(bin_stim_col_only_raw, ref, fs_plot, n_fft_ipd)

    ipd_ref_deg = np.rad2deg(ipd_ref)
    ipd_ctc_deg = np.rad2deg(ipd_ctc)
    ipd_filts_deg = np.rad2deg(ipd_filts)
    ipd_col_only_deg = np.rad2deg(ipd_col_only)
    err_ctc_deg = np.rad2deg(err_ctc)
    err_filts_deg = np.rad2deg(err_filts)
    err_col_only_deg = np.rad2deg(err_col_only)

    m_filts = mean_abs_ipd_error_deg(err_filts, freqs_ipd, *ipd_band)
    m_col = mean_abs_ipd_error_deg(err_col_only, freqs_ipd, *ipd_band)
    m_ctc = mean_abs_ipd_error_deg(err_ctc, freqs_ipd, *ipd_band)
    print(
        f"Mean |IPD error| in [{ipd_band[0]:.0f},{ipd_band[1]:.0f}] Hz (numpy, no bandpass) -- "
        f"Col.+IPD: {m_filts:.1f} deg, Col. only: {m_col:.1f} deg, CTC: {m_ctc:.1f} deg"
    )

    if ipd_model is not None:
        def _bin_torch(x):
            return torch.tensor(x, device=trainer.device, dtype=trainer.dtype).unsqueeze(0).unsqueeze(0)

        li = torch.tensor([centre_listener_idx], device=trainer.device)
        s_mod = ipd_model_score(ipd_model, _bin_torch(bin_stim_filts_raw), reference_stim, li)
        s_col = ipd_model_score(ipd_model, _bin_torch(bin_stim_col_only_raw), reference_stim, li)
        s_ctc = ipd_model_score(ipd_model, _bin_torch(bin_stim_ctc_raw), reference_stim, li)
        print(
            f"IPD module (rad, mean |err|) -- Col.+IPD: {s_mod:.4f}, Col. only: {s_col:.4f}, "
            f"CTC: {s_ctc:.4f}  (as deg: {np.rad2deg(s_mod):.1f}, {np.rad2deg(s_col):.1f}, {np.rad2deg(s_ctc):.1f})"
        )

    def _break_wraps(y, thresh=180.0):
        # Insert NaNs at +/-180 wrap discontinuities so the line is not drawn
        # across them (avoids misleading near-vertical connectors).
        y = y.astype(float).copy()
        jumps = np.abs(np.diff(y)) > thresh
        y[1:][jumps] = np.nan
        return y

    fig_ipd = plt.figure(figsize=(9, 4))

    axi1 = fig_ipd.add_subplot(1, 2, 1)
    axi1.axvspan(ipd_band[0], ipd_band[1], color="0.9", zorder=0)
    li1, = axi1.semilogx(freqs_ipd, ipd_filts_deg, linewidth=2.5)
    li1b, = axi1.semilogx(freqs_ipd, ipd_col_only_deg, linewidth=2.5)
    li2, = axi1.semilogx(freqs_ipd, ipd_ctc_deg, linewidth=2.5)
    li5, = axi1.semilogx(freqs_ipd, ipd_ref_deg, linewidth=3, color="black", linestyle="--")
    axi1.grid()
    axi1.set_xlim([100, 7000])
    axi1.set_ylim([-185, 185])
    axi1.set_title("Interaural Phase Difference (wrapped)", fontsize=fontsize_ipd)
    axi1.set_xlabel("Frequency [Hz]", fontsize=fontsize_ipd)
    axi1.set_ylabel("IPD [deg]", fontsize=fontsize_ipd)

    axi2 = fig_ipd.add_subplot(1, 2, 2)
    axi2.axvspan(ipd_band[0], ipd_band[1], color="0.9", zorder=0)
    axi2.semilogx(freqs_ipd, _break_wraps(err_filts_deg), linewidth=2.5)
    axi2.semilogx(freqs_ipd, _break_wraps(err_col_only_deg), linewidth=2.5)
    axi2.semilogx(freqs_ipd, _break_wraps(err_ctc_deg), linewidth=2.5)
    axi2.axhline(0.0, color="black", linestyle="--", linewidth=2)
    axi2.grid()
    axi2.set_xlim([100, 7000])
    axi2.set_ylim([-185, 185])
    axi2.set_title("IPD Error (vs Reference)", fontsize=fontsize_ipd)
    axi2.set_xlabel("Frequency [Hz]", fontsize=fontsize_ipd)
    axi2.set_ylabel("IPD error [deg]", fontsize=fontsize_ipd)

    fig_ipd.legend(
        [li1, li1b, li2, li5],
        ["Proposed Method (Col. + IPD)", "Proposed Method (Col. only)", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        fontsize=fontsize_ipd,
        frameon=False,
    )

    fig_ipd.tight_layout(rect=[0, 0, 1, 0.84])  # leave space for legend

    return fig, fig_phase, fig_ipd


def init_delta_fir(nn_model, n_samples):
    centre_tap_idx = n_samples // 2
    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[centre_tap_idx, 0] = 1.0


def _metric_at_centre(estimation_dict, key, centre_idx=0):
    if key not in estimation_dict or estimation_dict[key] is None:
        return np.nan
    val = np.asarray(estimation_dict[key]).reshape(-1)
    return float(val[centre_idx]) if val.size > centre_idx else float(np.mean(val))


def filter_lsd_to_ctc_db(est_filters, ctc_filts, fs=48000, eps=1e-12):
    """
    Log spectral distance (dB) between estimated and CTC loudspeaker filters.

    Matches the filter-domain LSD used in ``DAFx2026_fig7_plots`` (bandpassed
    magnitude spectra, RMS-normalised per filter, mean over channels).
    """
    est = bandpass_fir(np.asarray(est_filters, dtype=float), fs=fs)
    ctc = bandpass_fir(np.asarray(ctc_filts, dtype=float), fs=fs)
    est = est / np.sqrt(np.mean(est**2))
    ctc = ctc / np.sqrt(np.mean(ctc**2))
    n = est.shape[-1]

    est_mag_db = 20 * np.log10(np.abs(np.fft.rfft(est, n=n, axis=-1, norm="ortho")) + eps)
    ctc_mag_db = 20 * np.log10(np.abs(np.fft.rfft(ctc, n=n, axis=-1, norm="ortho")) + eps)
    lsd_per_channel = np.sqrt(np.mean((est_mag_db - ctc_mag_db) ** 2, axis=-1))
    return float(lsd_per_channel.mean())


def evaluate_fixed_filters(env, filters, eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader):
    """Evaluate precomputed loudspeaker filters with the same metrics as trained FIRs."""
    eval_prop = BinauralFreeFieldPropagation(
        fs=env.fs,
        lstn_pos=eval_lstn_pos,
        spk_pos=env.spk_pos,
        sofa_file_path=env.sofa_path,
        device=env.device,
        dtype=env.dtype,
    )
    filters = np.asarray(filters, dtype=float)
    filters = filters / (np.max(np.abs(filters)) / 0.9)

    listener_idx = torch.arange(len(eval_lstn_pos), device=env.device)
    estimation_dict = {key: [] for key in eval_perceptual_dict}

    with torch.no_grad():
        for stimulus in eval_stimulus_data_loader:
            filts_t = torch.tensor(filters, device=env.device, dtype=env.dtype).unsqueeze(0)
            bin_stim = eval_prop(stimulus, filts_t, listener_idx)
            for key, loss_cfg in eval_perceptual_dict.items():
                _, estimation = loss_cfg.compute_loss(
                    bin_stim=bin_stim,
                    spk_filters=filts_t,
                    stim=stimulus,
                    listener_indices=listener_idx,
                )
                if estimation is None:
                    continue
                estimation_dict[key].append(estimation.detach().cpu().numpy())

    return {
        key: np.mean(np.concatenate(vals, axis=0), axis=0)
        for key, vals in estimation_dict.items()
        if len(vals) > 0
    }


def run_single_seed(env, save_training_plots=True):
    """Train col+IPD and col-only; evaluate; return metrics and run directories."""
    env.seed = int(env.seed)
    set_seed(seed=env.seed)
    print(f"\n{'=' * 60}\nSeed {env.seed}\n{'=' * 60}")

    training_stimulus_data_loader, _ = get_stim_loader(
        env.fs, 1, 1, stimulus_type=env.selected_stim, device=env.device, dtype=env.dtype
    )
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
        normalize_log_spectrum=False, device=env.device, dtype=env.dtype,
    )
    ipd_model = interaural_phase_difference_matching(
        loss=F.l1_loss, target=0, target_pos=env.target_pos, fs=env.fs,
        sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype,
    )

    loss_dict = {
        "colouration": ListenerBasedLoss(model=col_model, weight=env.col_weight, device=env.device, dtype=env.dtype),
        "ipd": ListenerBasedLoss(model=ipd_model, weight=env.ipd_weight, device=env.device, dtype=env.dtype),
    }

    nn_model = MultiSourceFIR(n_samples=env.n_samples, n_sources=env.n_sources).to(device=env.device, dtype=env.dtype)
    init_delta_fir(nn_model, env.n_samples)
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
    t0 = time.time()
    trainer.train()
    time_col_ipd = time.time() - t0
    print(f"Time for col+IPD optimisation: {time_col_ipd:.2f} s")

    eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(
        env,
        eval_lstn_pos=torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5]], device=env.device, dtype=env.dtype),
    )
    eval_ipd_model = interaural_phase_difference_matching(
        loss=F.l1_loss, target=0, target_pos=env.target_pos, fs=env.fs,
        sofa_path=env.sofa_path, lstn_pos=eval_lstn_pos, device=env.device, dtype=env.dtype,
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

    saver_obj.save_results(estimation_dict=estimation_dict, est_filters=est_filters, fs=env.fs, env=env)

    if save_training_plots:
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
    init_delta_fir(nn_model_col_only, env.n_samples)
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
    t0 = time.time()
    trainer_col_only.train()
    time_col_only = time.time() - t0
    print(f"Time for colouration-only optimisation: {time_col_only:.2f} s")

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

    estimation_dict_ctc = evaluate_fixed_filters(
        env, CTC_filts, eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader,
    )

    eval_col_model = eval_perceptual_dict["colouration"].model
    colouration_loss_norm = float(eval_col_model.loss_norm)

    filter_lsd_to_ctc_col_ipd = filter_lsd_to_ctc_db(est_filters, CTC_filts, fs=env.fs)
    filter_lsd_to_ctc_col_only = filter_lsd_to_ctc_db(est_filters_col_only, CTC_filts, fs=env.fs)

    return {
        "seed": env.seed,
        "run_dir": saver_obj.run_dir,
        "run_dir_col_only": saver_obj_col_only.run_dir,
        "estimation_dict": estimation_dict,
        "estimation_dict_col_only": estimation_dict_col_only,
        "estimation_dict_ctc": estimation_dict_ctc,
        "trainer": trainer,
        "col_model": col_model,
        "eval_ipd_model": eval_ipd_model,
        "colouration_loss_norm": colouration_loss_norm,
        "filter_lsd_to_ctc_col_ipd": filter_lsd_to_ctc_col_ipd,
        "filter_lsd_to_ctc_col_only": filter_lsd_to_ctc_col_only,
        "CTC_filts": CTC_filts,
        "time_col_ipd": time_col_ipd,
        "time_col_only": time_col_only,
    }


def mean_ci_95(values, confidence=0.95):
    """
    Two-sided CI for the mean (Student-t), assuming independent runs.

    With n=1 the interval collapses to the single value. Requires n>=2 for a
    meaningful width; still well-defined for reporting when n_stat_seeds is small.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "ci_half_width": np.nan}
    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "ci_half_width": 0.0}
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    t_crit = float(scipy_stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half = t_crit * sem
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "ci_half_width": half}


def _stats_block(values, confidence=0.95, unit=None, description=None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    ci = mean_ci_95(arr, confidence=confidence)
    block = {
        "values": arr.tolist(),
        "mean": ci["mean"],
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "sem": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "ci_half_width": ci["ci_half_width"],
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if unit is not None:
        block["unit"] = unit
    if description is not None:
        block["description"] = description
    return block


def summarize_across_seeds(seed_results, centre_idx=0, confidence=0.95):
    """Aggregate centre-listener metrics across seeds (mean, std, sem, 95% CI)."""
    stats = {
        "n_seeds": len(seed_results),
        "seeds": [r["seed"] for r in seed_results],
        "centre_listener_idx": centre_idx,
        "confidence_level": confidence,
        "ci_note": (
            "CI is a two-sided Student-t interval for the mean across seeds "
            "(independent optimisation runs). Use n_stat_seeds >= 5 for stable CIs."
        ),
        "colouration_note": (
            "colouration_lsd_db is log spectral distance (LSD) in dB from spectral_matching. "
            "colouration_loss_normalised is LSD/loss_norm (the value used in the training loss)."
        ),
    }

    for method_key, est_key in (("col_ipd", "estimation_dict"), ("col_only", "estimation_dict_col_only")):
        lsd_values = []
        norm_values = []
        for r in seed_results:
            lsd = _metric_at_centre(r[est_key], "colouration", centre_idx)
            if not np.isfinite(lsd):
                continue
            lsd_values.append(lsd)
            norm_values.append(lsd / r["colouration_loss_norm"])

        block = _stats_block(
            lsd_values,
            confidence=confidence,
            unit="dB",
            description="Log spectral distance (LSD), centre listener",
        )
        if block is not None:
            stats[f"{method_key}_colouration_lsd_db"] = block

        block = _stats_block(
            norm_values,
            confidence=confidence,
            unit="dimensionless",
            description="Normalised colouration loss LSD/loss_norm",
        )
        if block is not None:
            stats[f"{method_key}_colouration_loss_normalised"] = block

        for metric in ("ipd", "localisation", "uncertainty"):
            values = [_metric_at_centre(r[est_key], metric, centre_idx) for r in seed_results]
            values = [v for v in values if np.isfinite(v)]
            block = _stats_block(values, confidence=confidence)
            if block is not None:
                if metric == "ipd":
                    block["unit"] = "rad"
                    block["description"] = "Mean absolute wrapped IPD error"
                stats[f"{method_key}_{metric}"] = block

    # CTC filters are deterministic; metrics are still listed per seed for consistency.
    ctc_lsd_values = []
    ctc_norm_values = []
    for r in seed_results:
        lsd = _metric_at_centre(r["estimation_dict_ctc"], "colouration", centre_idx)
        if not np.isfinite(lsd):
            continue
        ctc_lsd_values.append(lsd)
        ctc_norm_values.append(lsd / r["colouration_loss_norm"])

    block = _stats_block(
        ctc_lsd_values,
        confidence=confidence,
        unit="dB",
        description="Log spectral distance (LSD), centre listener, CTC filters",
    )
    if block is not None:
        stats["ctc_colouration_lsd_db"] = block

    block = _stats_block(
        ctc_norm_values,
        confidence=confidence,
        unit="dimensionless",
        description="Normalised colouration loss LSD/loss_norm, CTC filters",
    )
    if block is not None:
        stats["ctc_colouration_loss_normalised"] = block

    for metric in ("ipd", "localisation", "uncertainty"):
        values = [_metric_at_centre(r["estimation_dict_ctc"], metric, centre_idx) for r in seed_results]
        values = [v for v in values if np.isfinite(v)]
        block = _stats_block(values, confidence=confidence)
        if block is not None:
            if metric == "ipd":
                block["unit"] = "rad"
                block["description"] = "Mean absolute wrapped IPD error, CTC filters"
            stats[f"ctc_{metric}"] = block

    for method_key, result_key in (
        ("col_ipd", "filter_lsd_to_ctc_col_ipd"),
        ("col_only", "filter_lsd_to_ctc_col_only"),
    ):
        values = [r[result_key] for r in seed_results if np.isfinite(r[result_key])]
        block = _stats_block(
            values,
            confidence=confidence,
            unit="dB",
            description="LSD between loudspeaker filter magnitude spectra and CTC",
        )
        if block is not None:
            stats[f"{method_key}_filter_lsd_to_ctc_db"] = block

    return stats


def print_statistical_summary(stats):
    print(f"\n{'=' * 60}\nStatistical summary ({stats['n_seeds']} seeds, centre listener)\n{'=' * 60}")
    for method, label in (("col_ipd", "Col. + IPD"), ("col_only", "Col. only")):
        print(f"\n{label}:")
        rows = [
            ("colouration_lsd_db", "LSD [dB] (ear)"),
            ("filter_lsd_to_ctc_db", "LSD [dB] (filters vs CTC)"),
            ("colouration_loss_normalised", "Col. loss (norm.)"),
            ("ipd", "IPD [rad]"),
            ("localisation", "localisation"),
            ("uncertainty", "uncertainty"),
        ]
        pct = int(100 * stats.get("confidence_level", 0.95))
        for metric_key, metric_label in rows:
            key = f"{method}_{metric_key}"
            if key not in stats:
                continue
            s = stats[key]
            print(
                f"  {metric_label:22s}: {s['mean']:.4f}  "
                f"[{pct}% CI {s['ci_low']:.4f}, {s['ci_high']:.4f}]  "
                f"(std {s['std']:.4f}, n={len(s['values'])})"
            )

    print("\nCTC (analytic inversion, same filters every seed):")
    for metric_key, metric_label in rows:
        key = f"ctc_{metric_key}"
        if key not in stats:
            continue
        s = stats[key]
        print(
            f"  {metric_label:22s}: {s['mean']:.4f}  "
            f"[{pct}% CI {s['ci_low']:.4f}, {s['ci_high']:.4f}]  "
            f"(std {s['std']:.4f}, n={len(s['values'])})"
        )


def _stats_methods_for_metric(stats, metric_key):
    """Return (method_prefix, label, color) entries present in stats for this metric."""
    specs = [
        ("col_ipd", "Col. + IPD", "C0"),
        ("col_only", "Col. only", "C1"),
        ("ctc", "CTC", "C2"),
    ]
    return [(m, lab, col) for m, lab, col in specs if f"{m}_{metric_key}" in stats]


def plot_statistical_summary(stats, out_dir):
    """Bar charts: ear LSD, IPD, and filter LSD to CTC (95% CI over seeds)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("colouration_lsd_db", "Log spectral distance (ear)", "LSD [dB]"),
        ("ipd", "IPD error (ear)", "Mean |IPD error| [rad]"),
        ("filter_lsd_to_ctc_db", "LSD to CTC filters", "LSD [dB]"),
    ]
    metrics_to_plot = [
        spec for spec in plot_specs
        if len(_stats_methods_for_metric(stats, spec[0])) >= 2
    ]
    if not metrics_to_plot:
        return None

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    pct = int(100 * stats.get("confidence_level", 0.95))
    for ax, (metric_key, title, ylabel) in zip(axes, metrics_to_plot):
        methods = _stats_methods_for_metric(stats, metric_key)
        labels = [lab for _, lab, _ in methods]
        colors = [col for _, _, col in methods]
        x = np.arange(len(labels))
        means = [stats[f"{m}_{metric_key}"]["mean"] for m, _, _ in methods]
        ci_hw = [stats[f"{m}_{metric_key}"]["ci_half_width"] for m, _, _ in methods]
        ax.bar(x, means, 0.55, yerr=ci_hw, capsize=6, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        subtitle = f"n={stats['n_seeds']}, {pct}% CI"
        if metric_key == "colouration_lsd_db":
            subtitle = f"centre listener, {subtitle}"
        ax.set_title(f"{title}\n({subtitle})")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "DAFx2026_fig7_statistics.pdf", format="pdf")
    return fig


if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parent

    env = SimpleNamespace(
        target_angle=45.0,
        target_distance=100,
        epochs=1500,
        listener_batch_size=1,
        symmetric_listener_sampling=True,
        n_samples=256,
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.001,
        seed=46,
        log=1,
        fs=48000,
        # Set run_multi_seed_analysis=False for a single optimisation at env.seed only.
        run_multi_seed_analysis=True,
        n_stat_seeds=5,
        seed_start=46,
        save_training_plots=False,
        plot_fig7_on_last_seed=True,
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
    env.lstn_pos = torch.tensor(env.lstn_pos, device=env.device, dtype=env.dtype)
    env.spk_pos = torch.tensor(env.spk_pos, device=env.device, dtype=env.dtype)
    env.col_weight = 1.0
    env.ipd_weight = 1.0e-1

    if env.run_multi_seed_analysis:
        seeds = [env.seed_start + i for i in range(env.n_stat_seeds)]
        print(f"Multi-seed analysis: {env.n_stat_seeds} runs (seeds {seeds[0]}–{seeds[-1]})")
        seed_results = []
        for seed in seeds:
            env.seed = seed
            seed_results.append(
                run_single_seed(env, save_training_plots=env.save_training_plots)
            )

        stats = summarize_across_seeds(seed_results, centre_idx=0)
        print_statistical_summary(stats)

        last = seed_results[-1]
        last_run_dir = Path(last["run_dir"])
        plot_folder = last_run_dir / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)

        with open(last_run_dir / "seed_statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        plot_statistical_summary(stats, plot_folder)
        print(f"\nMulti-seed statistics saved in {last_run_dir}")
    else:
        print(f"Single-seed run (seed={env.seed}); skipping multi-seed statistics.")
        last = run_single_seed(env, save_training_plots=env.save_training_plots)
        last_run_dir = Path(last["run_dir"])
        plot_folder = last_run_dir / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)

    if env.plot_fig7_on_last_seed:
        CTC_bp = bandpass_fir(last["CTC_filts"], fs=env.fs)
        fig7_plot, fig7_phase_plot, fig7_ipd_plot = DAFx2026_fig7_plots(
            last["run_dir"],
            last["run_dir_col_only"],
            CTC_filts=CTC_bp,
            trainer=last["trainer"],
            col_model=last["col_model"],
            ipd_model=last["eval_ipd_model"],
        )
        fig7_plot.savefig(plot_folder / "DAFx2026_fig7.pdf", format="pdf")
        fig7_phase_plot.savefig(plot_folder / "DAFx2026_fig7_phase.pdf", format="pdf")
        fig7_ipd_plot.savefig(plot_folder / "DAFx2026_fig7_ipd.pdf", format="pdf")
        print(f"Fig.7 plots saved to {plot_folder}")


