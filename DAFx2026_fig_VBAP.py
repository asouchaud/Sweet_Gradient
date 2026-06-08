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
from scripts.helpers.filters import bandpass_fir


def run_experiment(loss_dict_builder, run_label, target_angle):
    repo_root = Path(__file__).resolve().parent
    d = 60
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=600,
        n_samples=21,
        target_pos=[d, float(np.tan(np.deg2rad(target_angle)) * d)],
        selected_stim="white_noise",
        listener_batch_size=1,
        stimulus_batch_size=1,
        n_stimulus=1,
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.01,
        seed=46,
        log=0,
        fs=48000,
    )
    env.dtype = torch.float32
    env.device_str =  "cpu"
    env.device = torch.device(env.device_str)

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    env.lstn_pos = [[0.0, 0.0]] 
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
    env.lstn_pos = torch.tensor(env.lstn_pos, device=env.device, dtype=env.dtype)
    env.spk_pos = torch.tensor(env.spk_pos, device=env.device, dtype=env.dtype)
    training_stimulus_data_loader, stimulus_dataset = get_stim_loader(
        env.fs,
        env.stimulus_batch_size,
        env.n_stimulus,
        stimulus_type=env.selected_stim,
        device=env.device,
        dtype=env.dtype,
    )
    training_listener_data_loader, _ = get_listener_loader(
        env.lstn_pos, env.listener_batch_size, symmetric=False
    )

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(fs=env.fs, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, sofa_file_path=env.sofa_path, device=env.device, dtype=env.dtype) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # Spectral Matching model
    unc_model = desena2020(loss = F.l1_loss, target = 0, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
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
        n_eval_stim=1,
    )
    est_filters, estimation_dict, bin_stim = trainer.evaluate(
        eval_lstn_pos=eval_lstn_pos,
        eval_stimulus_data_loader=eval_stimulus_data_loader,
        loss_dict=eval_perceptual_dict,
    )
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

    return(saver_obj.run_dir, env.lstn_pos, env.spk_pos)


def VBAP_loc(vbap_icld):

    repo_root = Path(__file__).resolve().parent
    fs = 48000
    dtype = torch.float32
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")

    lstn_pos = [[0.0, 0.0]] 
    base_angle_rad = np.deg2rad(30.0)
    speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
    spk_pos = [
        [speaker_distance * np.cos(base_angle_rad), speaker_distance * np.sin(base_angle_rad)],
        [speaker_distance * np.cos(base_angle_rad), -speaker_distance * np.sin(base_angle_rad)],
    ]

    lstn_pos = torch.tensor(lstn_pos, device=device, dtype=dtype)
    spk_pos = torch.tensor(spk_pos, device=device, dtype=dtype)
    reference_stim = torch.zeros(1, 1, int(0.01 * fs), device=device, dtype=dtype)
    reference_stim[0, 0, 0] = 1.0
    stereo_prop = BinauralFreeFieldPropagation(fs=fs, lstn_pos=lstn_pos, spk_pos=spk_pos, sofa_file_path=sofa_path, apply_distance_attenuation=True, device=device, dtype=dtype)
    loc_model = llado2025(loss = F.mse_loss, target_pos=[0,0], fs=fs, sofa_path=sofa_path, lstn_pos=lstn_pos, spk_pos=spk_pos, device=device, dtype=dtype) # Llado et al. 2025 model

    vbap_angle = torch.zeros(len(vbap_icld), device = device, dtype= dtype)
    vbap_icld = torch.tensor(vbap_icld, device=device, dtype=dtype)
    listener_idx = torch.arange(lstn_pos.shape[0], device=device)
    with torch.inference_mode():
        for i in range(len(vbap_icld)):

            right_gains = torch.sqrt(1/(1+ 10**(vbap_icld[i]/10)))
            left_gains = right_gains*10**(vbap_icld[i]/20)

            filts = torch.zeros(1,2,64,device=device, dtype=dtype) 
            filts[0,0,0] = left_gains
            filts[0,1,0] = right_gains

            bin_stim = stereo_prop(reference_stim, filts, listener_idx)
            vbap_angle[i] = loc_model(bin_stim).squeeze()

    return(vbap_angle.detach().cpu().numpy())


def DAFx2026_fig6_plots(run_folders, show=False):

    repo_root = Path(__file__).resolve().parent
    fs = 48000
    vbap_icld = np.loadtxt(repo_root / "data" / "miscellaneous" / "VBAP_panning_curve.csv")

    tmp = np.load(run_folders[0] / "artifacts" / "est_filters.npy")
    filt = np.zeros((13,2,tmp.shape[-1]))

    for i in range(len(run_folders)):
        filt[i] = np.load(run_folders[i] / "artifacts" / "est_filters.npy")

    filt_bp = bandpass_fir(filt, fs=fs)
    E_ratio = 10*np.log10(np.abs(np.sum(filt_bp[:,0,:]**2, axis = -1))/np.abs(np.sum(filt_bp[:,1,:]**2, axis = -1)))
    angles = np.arange(-30,35,5)

    r2 = r2_score(vbap_icld[1:-1], E_ratio[1:-1])

    # --- TOP PLOT ---
    plt.figure() 
    plt.subplot(1,2,1)
    l1, = plt.plot(angles[1:-1], -E_ratio[1:-1], linewidth=3)
    l2, = plt.plot(angles[1:-1], -vbap_icld[1:-1],  linewidth=2)
    plt.grid()
    plt.text(0.05, 0.95, f"$R^2 = {r2:.2f}$",
            transform=plt.gca().transAxes,
            fontsize=20,
            verticalalignment='top')
    plt.xlabel(r"Target Angle ($\theta^\circ$)", fontsize=18)
    plt.ylabel("Energy Ratio (dB)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # ax = plt.gca()
    # ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    target_angles = np.arange(30,-35,-5)
    vbap_angle = VBAP_loc(vbap_icld)
    est_angle = np.zeros(len(run_folders))
    for i in range(len(run_folders)):
        json_file_path = os.path.join(run_folders[i], "results.json")
        with open(json_file_path) as f:
            json_data = json.load(f)
        # One value per eval listener; use centre listener [0, 0] (index 0).
        est_angle[i] = float(np.asarray(json_data["localisation"], dtype=float).ravel()[0])

    vbap_mae = float(np.mean(np.abs(vbap_angle - target_angles)))
    est_mae = float(np.mean(np.abs(est_angle - target_angles)))
    print(f"Localisation MAE vs target angle:")
    print(f"  VBAP:             {vbap_mae:.2f} deg")
    print(f"  Estimated filters: {est_mae:.2f} deg")

    # --- BOTTOM PLOT (include +/- 30 deg; energy ratio is infinite there) ---
    plt.subplot(1,2,2)
    l3,= plt.plot(target_angles, est_angle, linewidth=3, label="Proposed Method")
    l4,= plt.plot(target_angles, vbap_angle, linewidth=2, label="VBAP")
    l5,= plt.plot(target_angles, target_angles, linestyle = "--", color = "black", linewidth=2, label = "Target")
    plt.xlabel(r"Target Angle ($\theta^\circ$)", fontsize=18)
    plt.ylabel(r"Predicted Angle ($\hat{\theta}^\circ$)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()


    # === LEGEND AT BOTTOM ===
    handles = [l3, l4, l5]
    labels = [h.get_label() for h in handles]

    plt.figlegend(handles, labels,
                loc='lower center',
                ncol=2,
                fontsize=20,
                bbox_to_anchor=(0.5, -0.02))

    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.25)
    plt.tight_layout(rect=[0, 0.16, 1, 1])  # leave space for legend
    plt.savefig("DAFx2026_fig6.pdf", format = "pdf")
    if show:
        plt.show()

    plt.figure(figsize=(6, 6))
    for i in range(7):
        right_gains = np.sqrt(1/(1+ 10**(vbap_icld[i+1]/10)))
        left_gains = right_gains*10**(vbap_icld[i+1]/20)

        vbap_filts = np.zeros((1,2,64)) 
        vbap_filts[0,0,0] = left_gains
        vbap_filts[0,1,0] = right_gains
        plt.subplot(7,2,2*i+1)

        plt.plot(filt[i+1,0], label = "Left Filter", linewidth = 4)
        plt.plot(vbap_filts[0,0], label = "VBAP Left Filter", linestyle = "--", linewidth = 2)
        plt.grid()    
        plt.ylim(-1.1,1.1)
        

        plt.subplot(7,2,2*i+2)
        plt.plot(filt[i+1,1], label = "Right Filter", linewidth = 4)
        plt.plot(vbap_filts[0,1], label = "VBAP Right Filter", linestyle = "--", linewidth = 2)           
        plt.grid()    
        plt.ylim(-1.1,1.1)

        print(f"VBAP Energy Ratio: {10*np.log10(np.sum(vbap_filts[0,0]**2)/np.sum(vbap_filts[0,1]**2)):.2f} dB")
        print(f"Estimated Energy Ratio: {10*np.log10(np.sum(filt[i+1,0]**2)/np.sum(filt[i+1,1]**2)):.2f} dB")

    if show:
        plt.show()
    

if __name__ == "__main__":
    angles = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, 355.0, 350.0, 345.0, 340.0, 335.0, 330.0]
    folders = []
    lstn_pos, spk_pos = None, None
    for angle in angles:
        run_dir, run_lstn_pos, run_spk_pos = run_experiment(
            loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype, angle=angle: {
                "localisation": ListenerBasedLoss(model=loc_model, weight=1e-0, device=device, dtype=dtype),
                "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e1),
                "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e1),
            },
            run_label=f"loc_{int(angle)}",
            target_angle=angle
        )
        folders.append(run_dir)
        if angle == 330.0:
            lstn_pos, spk_pos = run_lstn_pos, run_spk_pos
    
    DAFx2026_fig6_plots(folders)




    
