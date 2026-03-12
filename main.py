# Enable MPS fallback for PyTorch on macOS
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import interface libraries
import argparse
from pathlib import Path

# Import computational libraries
import torch
import torch.nn.functional as F

# Import libraries for plotting
import matplotlib.pyplot as plt
from scripts.plotting.graph_plot import plot_time
from scripts.plotting.graph_plot import plot_freq
from scripts.plotting.dense_arrows_plot import plot_arrows
from scripts.plotting.graph_plot import plot_color

# Import setup functions
from scripts.setup.setup import set_seed
from scripts.setup.setup import load_gtfb
from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs

# Import sound propagation module
from scripts.propagation.stereo_anechoic import stereo_anechoic

# Import perceptual models
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

# Import neural network related scripts
from scripts.optimisation.nn_init import load_NN
from scripts.optimisation.nn_init import load_optimiser
# from scripts.optimisation.nn_init import load_scheduler

# Import training script
from scripts.train.train import Train

# Import saving class
from scripts.saving.save_results import saver

# Import time
import time


def main(args):

## Initial setup

    # Set data type, device and sampling frequency
    dtype = torch.float32
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fs = 48000

    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(seed=args.seed)

    # Training parameters
    epochs = args.epochs # Number of training epochs

    # Loudspkeaker and listener positions
    lstn_pos_conf = [[[0.0, 0.0]],  # 1 listner position
                [[0.0, 0], [0.0, 0.5], [0.0, -0.5]], # 3 listener positions
                [[0.0, 0], [0.0, 0.5], [0.0, -0.5], [0.5, 0], [-0.5, 0]], # 5 listener positions
                [[0.0, 0], [0.0, 0.5], [0.0, -0.5], [0.5, 0], [-0.5, 0], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]] # 9 listener positions
    spk_pos = [[1.732,1], [1.732,-1]]

## Load data

    gtfb = load_gtfb(fs=fs, device=device, dtype=dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=fs, device=device, dtype=dtype, selected_stim=args.stimulus)  # Load stimulus
    hrirs = load_hrirs(dtype=dtype, device=device, sofa_path=args.sofa_path, idx_path=args.idx_path) # Load HRIRs using specific location from SOFA file

## Initialize model

    # Initialize sound propagation module
    stereo_prop = stereo_anechoic(stim=stim, lstn_pos=lstn_pos_conf[args.lstn_pos_conf], spk_pos=spk_pos, fs=fs, sofa_file_path=args.sofa_path) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_loss = F.mse_loss
    unc_loss = F.l1_loss
    col_loss = F.l1_loss

    loc_target = args.target_angle
    unc_target = 0
    col_target = 0

    if args.near_field == 1:
        loc_model = llado2025(loss = loc_loss, target = loc_target, hrirs=hrirs, fs=fs, gtfb=gtfb, lstn_pos_conf=lstn_pos_conf[args.lstn_pos_conf]).to(device=device, dtype=dtype) # Llado et al. 2025 model
        col_model = spectral_matching(loss = col_loss, target = col_target, stim=stim, hrirs=hrirs, target_angle=args.target_angle, fs=fs, lstn_pos_conf=lstn_pos_conf[args.lstn_pos_conf]).to(device=device, dtype=dtype) # Spectral Matching model
    else: 
        loc_model = llado2025(loss = loc_loss, target = loc_target, hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # Llado et al. 2025 model
        col_model = spectral_matching(loss = col_loss, target = col_target, stim=stim, hrirs=hrirs, target_angle=args.target_angle, fs=fs).to(device=device, dtype=dtype) # Spectral Matching model
    unc_model = desena2020(loss = unc_loss, target = unc_target, hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # De Sena et al. 2020 model
    

    perceptual_dict = {'localisation': loc_model,}
                         #'uncertainty': unc_model,}
                         #'colouration': col_model}

    # Initialise neural network
    nn_model = load_NN(n_samples=args.n_samples, device=device, dtype=dtype, init = args.init) # Load neural network ICLD or FIR
    nn_optimiser = load_optimiser(nn_model, lr=args.lr) # Load optimiser for neural network
    # nn_scheduler = load_scheduler(nn_optimiser, epochs)

## Initliase saving folder and logger

    saver_obj = saver(perceptual_dict, lstn_pos_conf[args.lstn_pos_conf], args)
    logger = saver_obj if args.log else None

## Training loop

    # Train the neural network
    trainer = Train(n_epoch=epochs,
                    nn=nn_model,
                    prop=stereo_prop,
                    perceptual_dict=perceptual_dict,
                    optimiser=nn_optimiser,
                    reg = args.reg,
                    loggers = [logger, None], 
                    scheduler=None,
                    gradnorm=args.gradnorm) # Initialize training class
    time_train_start = time.time()
    trainer.train()
    time_train = time.time() - time_train_start
    print(f"Time for full optimisation:{time_train:.2f}")
    
## Print and Plot results

    full_perceptual_dict = {'localisation': loc_model,
                         'uncertainty': unc_model,
                         'colouration': col_model}

    # Evaluate final neural network FIRs
    est_filters, estimation_dict, bin_stim = trainer.evaluate(full_perceptual_dict)

    # Print estimated perceptual metrics
    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    # Plot estimated filters (time and frequency domain)
    time_plot = plot_time(est_filters, show=False)
    freq_plot = plot_freq(est_filters, fs, show=False)
    arrow_plot = None
    if 'localisation' in perceptual_dict:
        if perceptual_dict['localisation'].lstn_pos_conf is not None:
            arrow_plot = plot_arrows(stereo_prop.lstn_pos, 
                                        stereo_prop.spk_pos, 
                                        torch.tensor(estimation_dict['localisation'], device=device, dtype=dtype), 
                                        torch.tensor(estimation_dict['uncertainty'], device=device, dtype=dtype) if 'uncertainty' in estimation_dict else None, 
                                        torch.tensor(estimation_dict['coloration'],device=device, dtype=dtype) if 'coloration' in estimation_dict else None, 
                                        loc_target=torch.tensor(args.target_angle),
                                        show=False
            )
        else:
            arrow_plot = plot_arrows(stereo_prop.lstn_pos, 
                                        stereo_prop.spk_pos, 
                                        torch.tensor(estimation_dict['localisation'], device=device,dtype=dtype), 
                                        torch.tensor(estimation_dict['uncertainty'],device=device,dtype=dtype) if 'uncertainty' in estimation_dict else None, 
                                        torch.tensor(estimation_dict['coloration'],device=device,dtype=dtype) if 'coloration' in estimation_dict else None, 
                                        show=False
            )
        color_l_plot, color_r_plot = plot_color(bin_stim, col_model.reference_fft, stereo_prop.lstn_pos)
        plt.show()

## Save results

    saver_obj.save_results(args, estimation_dict, est_filters, fs, lstn_pos_conf[args.lstn_pos_conf], spk_pos, trainer)
    saver_obj.save_plots(time_plot,freq_plot,arrow_plot, color_l_plot, color_r_plot)

if __name__ == "__main__":
    # Default paths relative to this repository (portable across machines/OSes)
    repo_root = Path(__file__).resolve().parent
    default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
    default_idx_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--near_field", type = int, default = 0, help = "Optimise for source in the near-field (1) and far-field (0)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--target_angle", type=float, default=00.0)
    parser.add_argument("--stimulus", type=str, default="random")
    parser.add_argument("--sofa_path", type=str, default=str(default_sofa_path))
    parser.add_argument("--idx_path", type=str,default=str(default_idx_path))
    parser.add_argument("--lstn_pos_conf", type=int, default=1, help="Listener position configuration index (0: 1 pos, 1: 3 pos, 2: 5 pos, 3: 9 pos)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--init", type=int, default=1, help="If 1, initialise NN with delta-like impulse responses, if 2 initialise with NN with delta-like impulse response at left speaker and 0s at right speaker")
    parser.add_argument("--reg", type=int, default=2, help="If 1, use first moment or 2 for second moment regularisation during training")
    parser.add_argument("--gradnorm", type=int, default=1, help="If 1, dynamically normalise weighting of losses using the GradNorm method")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default = 46, help="Random seed for reproducibility")
    parser.add_argument("--log", type=int, default=1, help="If 1, will log the training in Tensorboard, to view run in terminal | tensorboard --logdir runs | ")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computation (e.g., 'cpu', 'cuda', or 'mps'). If None, will use CUDA if available.")
    args = parser.parse_args()

    main(args)
