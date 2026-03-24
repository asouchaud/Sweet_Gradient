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
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train
from scripts.saving.save_results import saver


def run_experiment(loss_dict_builder, run_label, target_angle):
    repo_root = Path(__file__).resolve().parent
    d = 10
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=3000,
        num_lstn_pos=1,
        listener_batch_size=None,  # 5 mirrored pairs plus the centre listener
        symmetric_listener_sampling=True,
        n_samples=128,
        target_pos=[d, float(np.tan(np.deg2rad(target_angle)) * d)],
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.005,
        seed=46,
        log=1,
        fs=48000,
    )
    env.dtype = torch.float32
    env.device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env.device = torch.device(env.device_str)

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    env.lstn_pos = [[0.0, float(y)] for y in np.linspace(-0.5, 0.5, env.num_lstn_pos)] 
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
    unc_model = desena2020(loss = F.l1_loss, target = 0, hrirs=hrirs, fs=env.fs, gtfb=gtfb).to(device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
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

    return(saver_obj.run_dir, env.lstn_pos, env.spk_pos)



# ---------------------------
# Inversion
# ---------------------------
def compute_xtc_filters(H, N, reg=1e-3):
    # N = H.shape[-1]
    G = np.zeros_like(H, dtype=np.complex64)
    
    for k in range(N):
        Hf = H[:, :, k]
        G[:, :, k] = np.linalg.inv(Hf + reg * np.eye(2))
    
    return G


def sofa_find(source_angle, sofaile):
    pos = np.array(sofaile.SourcePosition[:, :2])
       
    source_angle = np.array([[source_angle, 0]])
    diff = source_angle[:, None, :] - pos[None, :, :]
    dists = np.linalg.vector_norm(diff, axis=2)
    return np.argmin(dists, axis=1)

def Compute_CTC_filters(target_angle):

    N = 128

    sofaile = sf.read_sofa("data\impulses\SOFA\D1_48K_24bit_256tap_FIR_SOFA.sofa")
    left_spk_hrirs = sofaile.Data_IR[760] # 30 deg
    right_spk_hrirs = sofaile.Data_IR[8064] # 330 deg
    ref_idx = sofa_find(target_angle, sofaile)
    ref_hrirs = np.squeeze(sofaile.Data_IR[ref_idx]) # 0 deg 
    ref_hrtf = np.fft.fft(ref_hrirs, n=N, axis = -1)

    IRs = np.concatenate((left_spk_hrirs[:,None,:], right_spk_hrirs[:,None,:]), axis = 1)
    H = np.fft.fft(IRs, n=N, axis = -1)
    # H, IRs = build_transfer_matrix()
    G = compute_xtc_filters(H, N)

    L_f = G[0,0] * ref_hrtf[0] + G[0,1] * ref_hrtf[1]
    R_f = G[1,0] * ref_hrtf[0] + G[1,1] * ref_hrtf[1]

    L = np.real(np.fft.ifft(L_f, axis = -1)).reshape(1,-1)
    R = np.real(np.fft.ifft(R_f, axis = -1)).reshape(1,-1)

    filts = np.concatenate((L,R), axis = 0)

    # ref_ear_L = causal_conv1d(stim,ref_hrirs[0])
    # ref_ear_R = causal_conv1d(stim,ref_hrirs[1])

    # spk_L, spk_R = apply_xtc(G, ref_hrirs[0], ref_hrirs[1])

    # ear_L = causal_conv1d(spk_L, left_spk_hrirs[0]) + causal_conv1d(spk_R, right_spk_hrirs[0])
    # ear_R = causal_conv1d(spk_L, left_spk_hrirs[1]) + causal_conv1d(spk_R, right_spk_hrirs[1])

    # tmp_L = causal_conv1d(impulse, filts[0])
    # tmp_R = causal_conv1d(impulse, filts[1])
    # test_L = causal_conv1d(tmp_L, left_spk_hrirs[0]) + causal_conv1d(tmp_R, right_spk_hrirs[0])
    # test_R = causal_conv1d(tmp_L, left_spk_hrirs[1]) + causal_conv1d(tmp_R, right_spk_hrirs[1])

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



def DAFx2026_fig7_plots(run_folder, CTC_filts):

    folder_name = run_folder[0]
    filt_path = os.path.join(folder_name, "artifacts", "est_filters.npy")
    filts = np.load(filt_path)
    N = 4096 #filts.shape[-1]

    ctc_rms = np.sqrt(np.mean(CTC_filts**2))
    filts_rms = np.sqrt(np.mean(filts**2))

    CTC_fft = np.fft.rfft(CTC_filts/ctc_rms, n = N, axis=-1)
    filts_fft = np.fft.rfft(filts/filts_rms, n = N, axis=-1)

    # Magnitude (already yours)
    CTC_f = 20*np.log10(np.abs(CTC_fft))
    filts_f = 20*np.log10(np.abs(filts_fft))

    # Phase
    CTC_phase = np.angle(CTC_fft)
    filts_phase = np.angle(filts_fft)

    CTC_phase_unwrapped = np.unwrap(CTC_phase, axis=-1)
    filts_phase_unwrapped = np.unwrap(filts_phase, axis=-1)

    freqs = np.fft.rfftfreq(N, d = 1/48000)

    CTC_f_smooth_left = smooth_third_octave(freqs, CTC_f[0])
    CTC_f_smooth_right = smooth_third_octave(freqs, CTC_f[1])
    filts_f_smooth_left = smooth_third_octave(freqs, filts_f[0])
    filts_f_smoot_right = smooth_third_octave(freqs, filts_f[1])

    fontsize = 20
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.semilogx(freqs, filts_f_smooth_left, label = "Proposed Method", linewidth = 3)
    plt.semilogx(freqs, CTC_f_smooth_left, label="CTC", linewidth = 3)
    plt.grid()
    plt.ylabel("Magnitude [dB]", fontsize = fontsize)
    plt.title("Left Loudspeaker Filter", fontsize = fontsize)
    plt.ylim([-25,20])
    plt.legend(fontsize = fontsize)
    plt.subplot(2,1,2)
    plt.title("Right Loudspeaker Filter", fontsize = fontsize)
    plt.semilogx(freqs, filts_f_smoot_right, label = "Proposed Method", linewidth = 3)
    plt.semilogx(freqs, CTC_f_smooth_right, label = "CTC", linewidth = 3)
    plt.grid()
    plt.legend(fontsize = fontsize)
    plt.xlabel("Frequency [Hz]", fontsize = fontsize)
    plt.ylabel("Magnitude [dB]", fontsize = fontsize)
    plt.ylim([-25,20])

    plt.tight_layout()
    plt.savefig("DAFx2026_fig7.pdf", format = "pdf")
    plt.show()




    

if __name__ == "__main__":

    target_angle = 0.0

    run_dir_col_0, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "colouration": ListenerBasedLoss(model=col_model, weight=1e-2, device=device, dtype=dtype)
        },
        run_label="col_0",
        target_angle=target_angle
    )

    CTC_filts = Compute_CTC_filters(target_angle=target_angle)

    
    folder = [run_dir_col_0]
    
    DAFx2026_fig7_plots(folder, CTC_filts=CTC_filts)




    
