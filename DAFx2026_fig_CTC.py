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

def smooth_third_octave(freq, mag_db):
    smoothed = np.zeros_like(mag_db)

    k = 2**(1/6)  # 1/3 octave half-width

    for i, fc in enumerate(freq):
        f_low = fc / k
        f_high = fc * k

        idx = np.where((freq >= f_low) & (freq <= f_high))[0]

        smoothed[i] = np.mean(mag_db[idx])

    return smoothed



def DAFx2026_fig7_plots(run_dir, run_dir_col_only, CTC_filts, trainer, col_model):

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


    fontsize = 18
    fig = plt.figure(figsize=(8,6))

    ax1 = plt.subplot(2,2,1)
    l1, = ax1.semilogx(freqs_filt, filts_f_smooth_left, label="Proposed Method", linewidth=3)
    l1b, = ax1.semilogx(freqs_filt, filts_col_only_f_smooth_left, label="Proposed Method (Col. only)", linewidth=3)
    l2, = ax1.semilogx(freqs_filt, CTC_f_smooth_left, label="CTC", linewidth=3)
    ax1.grid()
    ax1.set_ylabel("Magnitude [dB]", fontsize=fontsize)
    ax1.set_title("Left Loudspeaker Filter", fontsize=fontsize)
    ax1.set_xlim([80,20000])
    ax1.set_ylim([-30,10])

    ax2 = plt.subplot(2,2,2)
    ax2.set_title("Right Loudspeaker Filter", fontsize=fontsize)
    ax2.semilogx(freqs_filt, filts_f_smooth_right, linewidth=3)
    ax2.semilogx(freqs_filt, filts_col_only_f_smooth_right, linewidth=3)
    ax2.semilogx(freqs_filt, CTC_f_smooth_right, linewidth=3)
    ax2.grid()
    ax2.set_xlim([80,20000])
    ax2.set_ylim([-30,10])

    ax3 = plt.subplot(2,2,3)
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

    ax4 = plt.subplot(2,2,4)
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

    plt.tight_layout(rect=[0, 0, 1, 0.84])  # leave space for legend
    return fig



if __name__ == "__main__":
    
    repo_root = Path(__file__).resolve().parent
    
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        target_angle = 45.0,
        target_distance = 100,
        epochs=500,
        listener_batch_size=1,
        symmetric_listener_sampling=True,
        n_samples=256,
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.01,
        seed=46,
        log=1,
        fs=48000,
    )
    env.target_pos = [env.target_distance, float(np.tan(np.deg2rad(env.target_angle)) * env.target_distance)]
    
    env.dtype = torch.float32
    env.device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env.device = torch.device(env.device_str)

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    # env.lstn_pos = [[0.0, float(y)] for y in np.linspace(-0.5, 0.5, env.num_lstn_pos)] 
    # env.lstn_pos = [[0.0, float(i)] for i in np.arange(-0.06,0.07,0.03)] # Only the centre listener position is used for this experiment
    env.lstn_pos = [[0.0, 0.0]] # Only the centre listener position is used for this experiment
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
    training_stimulus_data_loader, stimulus_dataset = get_stim_loader(
        env.fs, 1, 1, stimulus_type=env.selected_stim, device=env.device, dtype=env.dtype
    )
    env.lstn_pos = torch.tensor(env.lstn_pos, device=env.device, dtype=env.dtype)
    env.spk_pos = torch.tensor(env.spk_pos, device=env.device, dtype=env.dtype)
    training_listener_data_loader, _ = get_listener_loader(
        env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling
    )

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(fs=env.fs, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, sofa_file_path=env.sofa_path, device=env.device, dtype=env.dtype) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, erb_weighting=False, normalize_log_spectrum=False, device=env.device, dtype=env.dtype) # Spectral Matching model
    ipd_model = interaural_phase_difference_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # Inter-aural Phase Difference model
    unc_model = desena2020(loss = F.l1_loss, target = 0, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
    regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)
    
    env.col_weight = 1.0e0
    env.ipd_weight = 1.0e-1

    loss_dict = {
        # "localisation": ListenerBasedLoss(model=loc_model, weight=env.loc_weight, device=env.device, dtype=env.dtype),
        # "uncertainty": ListenerBasedLoss(model=unc_model, weight=env.unc_weight, device=env.device, dtype=env.dtype),
        "colouration": ListenerBasedLoss(model=col_model, weight=env.col_weight, device=env.device, dtype=env.dtype),
        "ipd": ListenerBasedLoss(model=ipd_model, weight=env.ipd_weight, device=env.device, dtype=env.dtype),
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

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

## Initliase saving folder and logger
    saver_obj = saver()
    logger = saver_obj if env.log else None

## Training loop
    # Train the neural network
    trainer = Train(n_epoch=env.epochs,
                    stim_loader=training_stimulus_data_loader,
                    lstn_loader=training_listener_data_loader,
                    nn=nn_model,
                    prop=stereo_prop,
                    loss_dict=loss_dict,
                    optimiser=nn_optimiser,
                    logger=logger) # Initialize training class
    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

## Print and Plot results
    # Evaluate final neural network FIRs
    eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(
        env,
        eval_lstn_pos=torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5]], device=env.device, dtype=env.dtype),
    )
    eval_perceptual_dict["ipd"] = ListenerBasedLoss(model=interaural_phase_difference_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=eval_lstn_pos, device=env.device, dtype=env.dtype), weight=1, device=env.device, dtype=env.dtype)
    est_filters, estimation_dict, bin_stim = trainer.evaluate(eval_lstn_pos=eval_lstn_pos, eval_stimulus_data_loader=eval_stimulus_data_loader, loss_dict=eval_perceptual_dict)
    # Normalise filters to prevent clipping 
    est_filters /= np.max(np.abs(est_filters)) / 0.9 # TODO: move this to a more appropriate place at the end of the training loop

    # Print estimated perceptual metrics
    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    saver_obj.save_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        fs=env.fs,
        env=env,
    )
    bin_stim_for_plot = np.expand_dims(bin_stim, axis=0)
    plots = plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=bin_stim_for_plot,
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

    # ---------------------------
    # Colouration-only optimisation (for comparison curves)
    # ---------------------------
    set_seed(seed=env.seed)
    nn_model_col_only = MultiSourceFIR(
        n_samples=env.n_samples,
        n_sources=env.n_sources,
    ).to(device=env.device, dtype=env.dtype)
    with torch.no_grad():
        for branch in nn_model_col_only.branches:
            branch.weight.fill_(0.0)
            branch.weight[centre_tap_idx, 0] = 1.0

    nn_optimiser_col_only = torch.optim.Adam(nn_model_col_only.parameters(), lr=env.lr)

    loss_dict_col_only = {
        "colouration": ListenerBasedLoss(model=col_model, weight=env.col_weight, device=env.device, dtype=env.dtype),
    }

    saver_obj_col_only = saver()
    logger_col_only = saver_obj_col_only if env.log else None

    trainer_col_only = Train(n_epoch=env.epochs,
                    stim_loader=training_stimulus_data_loader,
                    lstn_loader=training_listener_data_loader,
                    nn=nn_model_col_only,
                    prop=stereo_prop,
                    loss_dict=loss_dict_col_only,
                    optimiser=nn_optimiser_col_only,
                    logger=logger_col_only)
    time_train_start = time.time()
    trainer_col_only.train()
    env.time_train_seconds_col_only = time.time() - time_train_start
    print(f"Time for colouration-only optimisation:{env.time_train_seconds_col_only:.2f}")

    est_filters_col_only, estimation_dict_col_only, bin_stim_col_only = trainer_col_only.evaluate(eval_lstn_pos=eval_lstn_pos, eval_stimulus_data_loader=eval_stimulus_data_loader, loss_dict=eval_perceptual_dict)
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
    
    fig7_plot = DAFx2026_fig7_plots(saver_obj.run_dir, saver_obj_col_only.run_dir, CTC_filts=CTC_filts, trainer=trainer, col_model=col_model)
    plot_folder = saver_obj.get_plot_folder()
    fig7_plot.savefig(plot_folder / "DAFx2026_fig7.pdf", format="pdf")


