# Import necessary python libraries
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

# Import libraries for plotting
from scripts.plotting.graph_plot import graph_plot

# Import setup functions
from scripts.setup.setup import set_seed
from scripts.setup.setup import load_gtfb
from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs
from scripts.setup.setup import load_template

# Import sound propagation module
from scripts.propagation.stereo_anechoic import stereo_anechoic

# Import perceptual models
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

# Import neural network related scripts
from scripts.optimisation.nn_init import load_NN
from scripts.optimisation.nn_init import load_optimiser

# Import training script
from scripts.train.train import Train


def main(args):
    
## Initial setup
    
    # Set data type, device and sampling frequency
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = 48000

    # Set random seed for reproducibility
    if args.seed:
        set_seed(seed=args.seed)

    # Training parameters
    epochs = args.epochs # Number of training epochs

    # Loudspkeaker and listener positions
    lstn_pos_conf = [[[-0.732, 0]],  # 1 listner position
                [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5]], # 3 listener positions
                [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5], [-0.232, 0], [-1.232, 0]], # 5 listener positions
                [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5], [-0.232, 0], [-1.232, 0], [-0.232, 0.5], [-0.232, -0.5], [-1.232, 0.5], [-1.232, -0.5]]] # 9 listener positions
    spk_pos = [[1,1], [1,-1]]

## Load data
    
    gtfb = load_gtfb(fs=fs, device=device, dtype=dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=fs, device=device, dtype=dtype, selected_stim=args.stimulus)  # Load stimulus
    hrirs = load_hrirs(dtype=dtype, device=device) # Load HRIRs using specific location from SOFA file 

## Initialize model

    # Initialize sound propagation module
    stereo_prop = stereo_anechoic(stim=stim, lstn_pos=lstn_pos_conf[0], spk_pos=spk_pos, fs=fs, sofa_file_path=args.sofa_path) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # Llado et al. 2025 model
    unc_model = desena2020(hrirs=hrirs, fs=fs, gtfb=gtfb).to(device=device, dtype=dtype) # De Sena et al. 2020 model
    col_model = spectral_matching(stim=stim, hrirs=hrirs, target_angle=0, fs=fs).to(device=device, dtype=dtype) # Spectral Matching model

    loc_loss = F.mse_loss
    unc_loss = F.mse_loss
    col_loss = F.l1_loss

    loc_target = args.target_angle
    unc_target = 0
    col_target = 0

    perceptual_dict = {'localisation': [loc_model, loc_target, loc_loss]} # Fill with model, target, loss function
                        #'uncertainty': [unc_model, unc_target, unc_loss],
                        #'coloration': [col_model, col_target, col_loss]}

    # Initialise neural network
    nn_model = load_NN(n_samples=args.n_samples, device=device, dtype=dtype, reg = args.init) # Load neural network ICLD or FIR
    nn_optimiser = load_optimiser(nn_model, lr=args.lr) # Load optimiser for neural network
    
## Training loop

    # Train the neural network
    trainer = Train(n_epoch=epochs,
                    nn=nn_model,
                    prop=stereo_prop,
                    perceptual_dict=perceptual_dict,
                    optimiser=nn_optimiser)
    
    trainer.train()

## Plot results

    # Evaluate final neural network FIRs
    
    est_filters = trainer.evaluate()

    # Plot estimated filters (time and frequency domain)
    graph_plot(est_filters, fs)

    

## Save results


if __name__ == "__main__":
    # Default paths relative to this repository (portable across machines/OSes)
    repo_root = Path(__file__).resolve().parent
    default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--target_angle", type=float, default=0.0)
    parser.add_argument("--stimulus", type=str, default="female_speech")
    parser.add_argument("--sofa_path", type=str, default=str(default_sofa_path))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--init", type=int, default=0, help="If 1, initialize NN with delta-like impulse responses")
    parser.add_argument("--seed", type=int, default = None, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args)
