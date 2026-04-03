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
    d = 60
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=300,
        num_lstn_pos=1,
        listener_batch_size=None,  # 5 mirrored pairs plus the centre listener
        symmetric_listener_sampling=False,
        n_samples=64,
        target_pos=[d, float(np.tan(np.deg2rad(target_angle)) * d)],
        selected_stim="impulse",
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"),
        lr=0.01,
        seed=46,
        log=1,
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


def VBAP_loc(vbap_icld):

    repo_root = Path(__file__).resolve().parent
    fs = 48000
    dtype = torch.float32
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    sofa_idx_path=str(repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv")

    lstn_pos = [[0.0, 0.0]] 
    base_angle_rad = np.deg2rad(30.0)
    speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
    spk_pos = [
        [speaker_distance * np.cos(base_angle_rad), speaker_distance * np.sin(base_angle_rad)],
        [speaker_distance * np.cos(base_angle_rad), -speaker_distance * np.sin(base_angle_rad)],
    ]

    gtfb = load_gtfb(fs=fs, device=device, dtype=dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=fs, device=device, dtype=dtype, selected_stim='impulse')  # Load stimulus
    hrirs = load_hrirs(dtype=dtype, device=device, sofa_path=sofa_path, idx_path=sofa_idx_path) # Load HRIRs using specific location from SOFA file

    stereo_prop = BinauralFreeFieldPropagation(stim=stim, lstn_pos=lstn_pos, spk_pos=spk_pos, fs=fs, sofa_file_path=sofa_path, apply_distance_attenuation=True)
    loc_model = llado2025(loss = F.mse_loss, target_pos=[0,0], hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # Llado et al. 2025 model

    vbap_angle = torch.zeros(len(vbap_icld), device = device, dtype= dtype)
    vbap_icld = torch.tensor(vbap_icld, device=device, dtype=dtype)
    for i in range(len(vbap_icld)):

        right_gains = torch.sqrt(1/(1+ 10**(vbap_icld[i]/10)))
        left_gains = right_gains*10**(vbap_icld[i]/20)

        filts = torch.zeros(1,2,64,device=device, dtype=dtype) 
        filts[0,0,0] = left_gains
        filts[0,1,0] = right_gains

        bin_stim = stereo_prop(filts)

        vbap_angle[i] = loc_model(bin_stim)

    return(vbap_angle.detach().cpu().numpy())


def DAFx2026_fig6_plots(run_folders):

    repo_root = Path(__file__).resolve().parent
    vbap_icld = np.loadtxt(repo_root / "data" / "miscellaneous" / "VBAP_panning_curve.csv")

    tmp = np.load(run_folders[0] / "artifacts" / "est_filters.npy")
    filt = np.zeros((13,2,tmp.shape[-1]))

    for i in range(len(run_folders)):
        filt[i] = np.load(run_folders[i] / "artifacts" / "est_filters.npy")

    E_ratio = 10*np.log10(np.abs(np.sum(filt[:,0,:]**2, axis = -1))/np.abs(np.sum(filt[:,1,:]**2, axis = -1)))
    angles = np.arange(-30,35,5)

    r2 = r2_score(vbap_icld[1:-1], E_ratio[1:-1])

    # --- TOP PLOT ---
    plt.figure(figsize=(6, 7)) 
    plt.subplot(2,1,1)
    l1, = plt.plot(angles[1:-1], -E_ratio[1:-1], linewidth=3)
    l2, = plt.plot(angles[1:-1], -vbap_icld[1:-1],  linewidth=2)
    plt.grid()
    plt.text(0.05, 0.95, f"$R^2 = {r2:.3f}$",
            transform=plt.gca().transAxes,
            fontsize=20,
            verticalalignment='top')
    plt.ylabel("Energy Ratio (dB)", fontsize=20)
    plt.yticks(fontsize=20)
    ax = plt.gca()
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    target_angles = np.arange(30,-35,-5)
    vbap_angle = VBAP_loc(vbap_icld)
    est_angle = np.zeros(len(run_folders))
    for i in range(len(folders)):
        json_file_path = os.path.join(run_folders[i] /  "results.json")
        json_data = json.load(open(json_file_path))

        est_angle[i] = json_data['localisation']

    # --- BOTTOM PLOT ---
    plt.subplot(2,1,2)
    l3,= plt.plot(target_angles[1:-1], est_angle[1:-1], linewidth=3, label="Proposed Method")
    l4,= plt.plot(target_angles[1:-1], vbap_angle[1:-1], linewidth=2, label="VBAP")
    l5,= plt.plot(target_angles[1:-1], target_angles[1:-1], linestyle = "--", color = "black", linewidth=2, label = "Target")
    plt.xlabel(r"Target Angle ($\theta^\circ$)", fontsize=20)
    plt.ylabel(r"Predicted Angle ($\hat{\theta}^\circ$)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.legend(fontsize=15)


    # # === LEGEND AT BOTTOM ===
    # handles = [l1, l2, l5]
    # labels = [h.get_label() for h in handles]

    # plt.figlegend(handles, labels,
    #             loc='lower center',
    #             ncol=2,
    #             fontsize=20,
    #             bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.savefig("DAFx2026_fig6.pdf", format = "pdf")
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

    plt.show()
    

if __name__ == "__main__":
    run_dir_loc_30, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_30",
        target_angle=30.0
    )

    run_dir_loc_25, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_25",
        target_angle=25.0
    )

    run_dir_loc_20, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_20",
        target_angle=20.0
    )

    run_dir_loc_15, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_15",
        target_angle=15.0
    )

    run_dir_loc_10, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_10",
        target_angle=10.0
    )

    run_dir_loc_5, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_5",
        target_angle=5.0
    )

    run_dir_loc_0, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_0",
        target_angle=0.0
    )

    run_dir_loc_355, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_355",
        target_angle=355.0
    )

    run_dir_loc_350, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_350",
        target_angle=350.0
    )

    run_dir_loc_345, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_345",
        target_angle=345.0
    )

    run_dir_loc_340, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_340",
        target_angle=340.0
    )

    run_dir_loc_335, _, _ = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e1, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_335",
        target_angle=335.0
    )

    run_dir_loc_330, lstn_pos, spk_pos = run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1e-0, device=device, dtype=dtype),
            "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1e-0),
            "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1e-0),
        },
        run_label="loc_330",
        target_angle=330.0
    )

    folders = [run_dir_loc_30,
               run_dir_loc_25,
               run_dir_loc_20,
               run_dir_loc_15,
               run_dir_loc_10,
               run_dir_loc_5,
               run_dir_loc_0,
               run_dir_loc_355,
               run_dir_loc_350,
               run_dir_loc_345,
               run_dir_loc_340,
               run_dir_loc_335,
               run_dir_loc_330]
    
    DAFx2026_fig6_plots(folders)




    
