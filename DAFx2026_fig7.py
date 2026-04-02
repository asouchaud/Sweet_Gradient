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
from sklearn.metrics import r2_score

# Import Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors


# Import project modules
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.plotting.results_plot import plot_results
from scripts.setup.setup import load_gtfb, load_hrirs, load_stim, set_seed
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.perceptual_models.ipd.ipd import interaural_phase_difference
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
from scripts.saving.save_results import saver




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



def DAFx2026_fig7_plots(run_dir, CTC_filts, trainer, col_model):

    folder_name = run_dir
    filt_path = os.path.join(folder_name, "artifacts", "est_filters.npy")
    filts = np.load(filt_path)
    N_filts =  filts.shape[-1]

    ctc_normed = CTC_filts / np.sqrt(np.mean(CTC_filts**2))
    filts_normed = filts / np.sqrt(np.mean(filts**2))

    ctc_filts_torch = torch.tensor(ctc_normed, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    filts_torch = torch.tensor(filts_normed, device=trainer.device, dtype=trainer.dtype).unsqueeze(0)
    
    bin_stim_ctc = trainer.prop(ctc_filts_torch).cpu().numpy()[0].squeeze()
    bin_stim_filts = trainer.prop(filts_torch).cpu().numpy()[0].squeeze()
    
    ref = col_model.reference_stim.cpu().numpy()
    N_ref =  ref.shape[-1]

    bin_stim_ctc_normed = bin_stim_ctc / np.sqrt(np.mean(bin_stim_ctc**2))
    bin_stim_filts_normed = bin_stim_filts / np.sqrt(np.mean(bin_stim_filts**2))
    ref_normed = ref / np.sqrt(np.mean(ref**2))

    CTC_fft = np.fft.rfft(ctc_normed, n = N_filts, axis=-1)
    filts_fft = np.fft.rfft(filts_normed, n = N_filts, axis=-1)

    # Magnitude
    CTC_f = 20*np.log10(np.abs(CTC_fft))
    filts_f = 20*np.log10(np.abs(filts_fft))

    lsd_freq = np.sqrt(np.mean((CTC_f - filts_f) ** 2, axis=-1))
    lsd = lsd_freq.mean(axis=-1)
    print(f"Log Spectral Distance between CTC and proposed method filters: {lsd:.2f} dB")

    freqs_filt = np.fft.rfftfreq(N_filts, d = 1/48000)

    CTC_f_smooth_left = smooth_third_octave(freqs_filt, CTC_f[0])
    CTC_f_smooth_right = smooth_third_octave(freqs_filt, CTC_f[1])
    filts_f_smooth_left = smooth_third_octave(freqs_filt, filts_f[0])
    filts_f_smooth_right = smooth_third_octave(freqs_filt, filts_f[1])


    ref_fft = np.fft.rfft(ref_normed, n = N_ref, axis=-1)
    bin_stim_filts_fft = np.fft.rfft(bin_stim_filts_normed, n = N_ref, axis=-1)
    bin_stim_ctc_fft = np.fft.rfft(bin_stim_ctc_normed, n = N_ref, axis=-1)
    
    ref_f = 20*np.log10(np.abs(ref_fft))
    bin_stim_filts_f = 20*np.log10(np.abs(bin_stim_filts_fft))
    bin_stim_ctc_f = 20*np.log10(np.abs(bin_stim_ctc_fft))

    freqs_ref = np.fft.rfftfreq(N_ref, d = 1/48000)

    ref_f_smooth_left = smooth_third_octave(freqs_ref, ref_f[0,0])
    ref_f_smooth_right = smooth_third_octave(freqs_ref, ref_f[0,1])
    bin_stim_ctc_f_smooth_left = smooth_third_octave(freqs_ref, bin_stim_ctc_f[0])
    bin_stim_ctc_f_smooth_right = smooth_third_octave(freqs_ref, bin_stim_ctc_f[1])
    bin_stim_filts_f_smooth_left = smooth_third_octave(freqs_ref, bin_stim_filts_f[0])
    bin_stim_filts_f_smooth_right = smooth_third_octave(freqs_ref, bin_stim_filts_f[1])


    fontsize = 20
    fig = plt.figure(figsize=(8,6))

    ax1 = plt.subplot(2,2,1)
    l1, = ax1.semilogx(freqs_filt, filts_f_smooth_left, label="Proposed Method", linewidth=3)
    l2, = ax1.semilogx(freqs_filt, CTC_f_smooth_left, label="CTC", linewidth=3)
    ax1.grid()
    ax1.set_ylabel("Magnitude [dB]", fontsize=fontsize)
    ax1.set_title("Left Loudspeaker Filter", fontsize=fontsize)
    ax1.set_ylim([0,35])

    ax2 = plt.subplot(2,2,2)
    ax2.set_title("Right Loudspeaker Filter", fontsize=fontsize)
    ax2.semilogx(freqs_filt, filts_f_smooth_right, linewidth=3)
    ax2.semilogx(freqs_filt, CTC_f_smooth_right, linewidth=3)
    ax2.grid()
    ax2.set_ylim([0,35])

    ax3 = plt.subplot(2,2,3)
    l3, = ax3.semilogx(freqs_ref, bin_stim_filts_f_smooth_left, linewidth=3)
    l4, = ax3.semilogx(freqs_ref, bin_stim_ctc_f_smooth_left, linewidth=3)
    l5, = ax3.semilogx(freqs_ref, ref_f_smooth_left, label="Reference", linewidth=4, color="black", linestyle="--")
    ax3.grid()
    ax3.set_ylim([0,35])
    ax3.set_title("Left Ear Signal", fontsize=fontsize)
    ax3.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax3.set_ylabel("Magnitude [dB]", fontsize=fontsize)

    ax4 = plt.subplot(2,2,4)
    ax4.semilogx(freqs_ref, bin_stim_filts_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_ref, bin_stim_ctc_f_smooth_right, linewidth=3)
    ax4.semilogx(freqs_ref, ref_f_smooth_right, linewidth=4, color="black", linestyle="--")
    ax4.grid()
    ax4.set_ylim([0,35])
    ax4.set_title("Right Ear Signal", fontsize=fontsize)
    ax4.set_xlabel("Frequency [Hz]", fontsize=fontsize)

    # ONE common legend
    fig.legend(
        [l1, l2, l5],
        ["Proposed Method", "CTC", "Reference"],
        loc="upper center",
        ncol=3,
        fontsize=fontsize
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for legend
    return fig



if __name__ == "__main__":
    target_angle = 15.0

    repo_root = Path(__file__).resolve().parent
    target_distance = 100
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=1000,
        listener_batch_size=None,  # 5 mirrored pairs plus the centre listener
        symmetric_listener_sampling=True,
        n_samples=256,
        target_pos=[target_distance, float(np.tan(np.deg2rad(target_angle)) * target_distance)],
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.01,
        seed=48,
        log=1,
        fs=48000,
    )
    env.dtype = torch.float32
    env.device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env.device = torch.device(env.device_str)

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    # env.lstn_pos = [[0.0, float(y)] for y in np.linspace(-0.5, 0.5, env.num_lstn_pos)] 
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
    gtfb = load_gtfb(fs=env.fs, device=env.device, dtype=env.dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=env.fs, device=env.device, dtype=env.dtype, selected_stim=env.selected_stim)  # Load stimulus
    hrirs = load_hrirs(dtype=env.dtype, device=env.device, sofa_path=env.sofa_path, idx_path=env.sofa_idx_path) # Load HRIRs using specific location from SOFA file

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(stim=stim, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, fs=env.fs, sofa_file_path=env.sofa_path) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, stim=stim, hrirs=hrirs, target_pos=env.target_pos, fs=env.fs).to(device=env.device, dtype=env.dtype) # Spectral Matching model
    ipd_model = interaural_phase_difference(loss = F.l1_loss, target = 0, reference_stim=col_model.reference_stim, fs=env.fs).to(device=env.device, dtype=env.dtype) # Inter-aural Phase Difference model
    unc_model = desena2020(loss = F.l1_loss, target = 0, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
    regularisation = second_moment_regularisation(n_samples=env.n_samples, device=env.device, dtype=env.dtype)
    
    env.col_weight = 1.0e0
    env.ipd_weight = 1.0e0

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
            "colouration": ListenerBasedLoss(model=col_model, weight=1, device=env.device, dtype=env.dtype),
            "ipd": ListenerBasedLoss(model=ipd_model, weight=1, device=env.device, dtype=env.dtype),
        }
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
    saver_obj.save_plots(plots)

    # Save per-epoch source FIR evolution as videos (one per source)
    # saver_obj.save_filter_video(trainer.h_history, source_index=0, fs=env.fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=1, fs=env.fs)

    plt.close('all')

    CTC_filts = Compute_CTC_filters(target_angle=target_angle)
    
    fig7_plot = DAFx2026_fig7_plots(saver_obj.run_dir, CTC_filts=CTC_filts, trainer=trainer, col_model=col_model)
    plot_folder = saver_obj.get_plot_folder()
    fig7_plot.savefig(plot_folder / "DAFx2026_fig7.pdf", format="pdf")


