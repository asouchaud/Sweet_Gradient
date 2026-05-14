# Enable MPS fallback for PyTorch on macOS
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
_cpu_count = os.cpu_count() or 1
os.environ.setdefault('OMP_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('MKL_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(_cpu_count))

# Import interface libraries
from pathlib import Path
import time
from types import SimpleNamespace

# Import computational libraries
import numpy as np
import torch
import torch.nn.functional as F

# Import project modules
from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.datasets.listener_dataset import get_listener_loader

from scripts.optimisation.custom_modules import MultiSourceFIR

from scripts.plotting.results_plot import plot_results
from scripts.plotting.itd_ild_animation import save_itd_ild_evolution_animation
from scripts.plotting.combined_distance_animation import save_combined_distance_evolution_animation

from scripts.setup.setup import set_seed, build_eval_objects

from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation

from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.perceptual_models.ipd.ipd import interaural_phase_difference_matching

from scripts.optimisation.regularisation import second_moment_regularisation

from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train

from scripts.saving.save_results import saver


def run_experiment(loss_dict_builder, run_label):
    repo_root = Path(__file__).resolve().parent
    d = 100
    fs = 48000
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=100,
        num_lstn_pos=1,
        listener_batch_size=1,  # 5 mirrored pairs plus the centre listener
        symmetric_listener_sampling=False,
        selected_stim="impulse",
        stimulus_batch_size = 1,
        n_stimulus = 1,
        n_filter_taps=32,
        target_pos=[d, float(np.tan(np.deg2rad(15.0)) * d)],
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        lr=0.1,
        seed=46,
        log=1,
        fs=fs,
    )
    # Backwards-compatible alias for helpers that still expect `n_samples`.
    env.n_samples = env.n_filter_taps
    env.dtype = torch.float64
    env.device_str = "cpu"
    env.device = torch.device(env.device_str)
    env.num_threads = _cpu_count
    env.num_interop_threads = min(8, _cpu_count)

    # Maximize CPU-side parallelism for PyTorch ops.
    torch.set_num_threads(env.num_threads)
    torch.set_num_interop_threads(env.num_interop_threads)
    torch.backends.mkldnn.enabled = True

    assert isinstance(env.listener_batch_size, int), "listener_batch_size must be set to positive integer"
    assert env.listener_batch_size > 0 , "listener_batch_size must be set to positive integer"

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    # env.lstn_pos = torch.tensor([[0.0, float(y)] for y in np.linspace(-0.5, 0.5, env.num_lstn_pos)], 
    #                             device = env.device, 
    #                             dtype = env.dtype)
    env.lstn_pos = torch.tensor([[0.0, 0.0]], device = env.device, dtype = env.dtype)
    base_angle_rad = np.deg2rad(30.0)
    env.speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
    env.spk_pos = [
        [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
        [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
    ]
    env.spk_pos = torch.tensor(env.spk_pos, device = env.device, dtype = env.dtype)
    env.n_sources = len(env.spk_pos)

    # Reset the seed so each run starts from the same point.
    set_seed(seed=env.seed)

## Load data
    training_stimulus_data_loader, stimulus_dataset = get_stim_loader(env.fs, env.stimulus_batch_size, env.n_stimulus, stimulus_type=env.selected_stim, device=env.device, dtype=env.dtype) # Get stimulus dataloader
    training_listener_data_loader, _ = get_listener_loader(env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling) # Get listener dataloader

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(fs=env.fs, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, sofa_file_path=env.sofa_path, device=env.device, dtype=env.dtype) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # Spectral Matching model
    unc_model = desena2020(loss = F.l1_loss, target = 0, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # De Sena et al. 2020 model
    ipd_model = interaural_phase_difference_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # Interaural Phase Difference model
    regularisation = second_moment_regularisation(n_samples=env.n_filter_taps, device=env.device, dtype=env.dtype)
    loss_dict = loss_dict_builder(loc_model, col_model, unc_model, regularisation, env.device, env.dtype)

    # Initialise neural network
    nn_model = MultiSourceFIR(
        n_samples=env.n_filter_taps,
        n_sources=env.n_sources,
    ).to(device=env.device, dtype=env.dtype)
    # Initialise FIR filter weights
    centre_tap_idx = env.n_filter_taps // 2
    with torch.no_grad():
        for i, branch in enumerate(nn_model.branches):
            branch.weight.fill_(0.0)
            branch.weight[centre_tap_idx] = 1.0
        # nn_model.branches[0].weight[0, 0] = 1.0

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

## Initliase saving folder and logger
    saver_obj = saver()
    logger = saver() if env.log else None

## Training loop
    # Train the neural network
    trainer = Train(n_epoch=env.epochs,
                    stim_loader = training_stimulus_data_loader,
                    lstn_loader = training_listener_data_loader,
                    nn=nn_model,
                    prop=stereo_prop,
                    loss_dict=loss_dict,
                    optimiser=nn_optimiser,
                    logger=logger
                    ) # Initialize training class
    

    print("Evaluating perceptual attributes and loss before training (initialised filters)...")
    _, initial_estimation_dict, _ = trainer.evaluate(
        eval_lstn_pos=env.lstn_pos,
        eval_stimulus_data_loader=training_stimulus_data_loader,
        loss_dict=loss_dict,
    )
    for key, value in initial_estimation_dict.items():
        print(f"Initial estimated {key}: {value}")

## TRAINING LOOP
    time_train_start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - time_train_start
    print(f"Time for full optimisation:{env.time_train_seconds:.2f}")

## Print and Plot results
    # Evaluate final neural network FIRs
    # eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(env, eval_lstn_pos=env.lstn_pos)
    est_filters, estimation_dict, bin_stim = trainer.evaluate(eval_lstn_pos=env.lstn_pos, eval_stimulus_data_loader=training_stimulus_data_loader, loss_dict=loss_dict)
    
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
        loss_dict=loss_dict,
        localisation_tracking=trainer.localisation_tracking,
    )

    if trainer.localisation_tracking is not None:
        save_itd_ild_evolution_animation(
            localisation_tracking=trainer.localisation_tracking,
            output_path=saver_obj.run_dir / "plots" / "itd_ild_evolution.gif",
            fps=10,
        )
        save_combined_distance_evolution_animation(
            localisation_tracking=trainer.localisation_tracking,
            output_path=saver_obj.run_dir / "plots" / "combined_distance_evolution.gif",
            fps=10,
        )

    # plots = plot_results(
    #     estimation_dict=estimation_dict,
    #     est_filters=est_filters,
    #     bin_stim=bin_stim,
    #     lstn_pos=env.lstn_pos,
    #     spk_pos=env.spk_pos,
    #     loss_dict=eval_perceptual_dict,
    #     target_pos=env.target_pos,
    #     fs=env.fs,
    #     stim = eval_stimulus_dataset.data,
    #     device=env.device,
    #     dtype=env.dtype,
    #     show=True,
    # )
    # saver_obj.save_plots(plots)

    # Save per-epoch source FIR evolution as videos (one per source)
    # saver_obj.save_filter_video(trainer.h_history, source_index=0, fs=env.fs)
    # saver_obj.save_filter_video(trainer.h_history, source_index=1, fs=env.fs)


if __name__ == "__main__":
    run_experiment(
        loss_dict_builder=lambda loc_model, col_model, unc_model, regularisation, device, dtype: {
            "localisation": ListenerBasedLoss(model=loc_model, weight=1.0e0, device=device, dtype=dtype),
            "colouration": ListenerBasedLoss(model=col_model, weight=1.0e-3, device=device, dtype=dtype),
            "uncertainty": ListenerBasedLoss(model=unc_model, weight=1.0e-3, device=device, dtype=dtype),
            # "regularisation_source_0": SourceBasedLoss(regulariser=regularisation, source_idx=0, weight=1.0e-3),
            # "regularisation_source_1": SourceBasedLoss(regulariser=regularisation, source_idx=1, weight=1.0e-3),
        },
        run_label="localisation_only",
    )
