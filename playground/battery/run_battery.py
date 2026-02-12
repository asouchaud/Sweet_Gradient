# Import interface libraries
import argparse
from pathlib import Path

# Import computational libraries
import torch
import torch.nn.functional as F

# Import libraries for plotting

# Import setup functions
from playground.battery.scripts.setup.setup import setup
from playground.battery.scripts.setup.setup import get_all_subsets
from playground.battery.scripts.setup.setup import init_perc_model

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

# Import saving functions
from playground.battery.scripts.saving.save_results import saver

from datetime import datetime


def run_battery(args):

    # Get current datetime
    now = datetime.now()

    # Extract date and time separately
    current_date = now.strftime("%y%m%d") # YYMMDD
    current_time = now.strftime("%H%M%S")  # HHMMSS

    print("Date:", current_date)
    print("Time:", current_time)

    # Initial setup
    dtype, device, fs, target_angle, lstn_pos_conf, spk_pos, gtfb, stim, hrirs = setup(args)

    repo_root = Path(__file__).resolve().parent
    default_sofa_path = repo_root / ".."/ ".." /"data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"

    n_ta = target_angle.shape[0]
    n_lstn_pos_conf = len(lstn_pos_conf)

    full_dict = [{'localisation': llado2025}, {'uncertainty': desena2020}, {'colouration': spectral_matching}]
    dict_subsets = get_all_subsets(full_dict)
    
    for lstn_pos_conf_idx in range(n_lstn_pos_conf):
        for ta_idx in range(n_ta):
            for dict_sub_idx in range(7): # 7 different initialisation and regularisation combinations
                print(f"Running battery for target angle {target_angle[ta_idx].item()} deg, listener position configuration {lstn_pos_conf_idx}, current run {dict_sub_idx+1}/7")
                
                # Initialize sound propagation module
                stereo_prop = stereo_anechoic(stim=stim,
                                 lstn_pos=lstn_pos_conf[lstn_pos_conf_idx],
                                 spk_pos=spk_pos,
                                 fs=fs,
                                 sofa_file_path=default_sofa_path) # Load stereo anechoic propagation module
                
                # Initialize model
                perceptual_dict = init_perc_model(dict_subsets[dict_sub_idx], device, dtype, fs, stim, hrirs, target_angle[ta_idx], gtfb)
                
                # Initialise neural network
                nn_model = load_NN(n_samples=args.n_samples, device=device, dtype=dtype, init = args.init) # Load neural network ICLD or FIR
                nn_optimiser = load_optimiser(nn_model, lr=0.001) # Load optimiser for neural network

                saver_obj = saver(current_date=current_date,
                                  current_time=current_time,
                                  reg_idx=args.reg, 
                                  init_idx=args.init, 
                                  n_samples=args.n_samples, 
                                  angle=target_angle[ta_idx], 
                                  lstn_idx=lstn_pos_conf_idx,
                                  curr_idx=dict_sub_idx)
                logger = saver_obj if args.log else None

                # Train the neural network
                trainer = Train(n_epoch=args.epochs,
                    nn=nn_model,
                    prop=stereo_prop,
                    perceptual_dict=perceptual_dict,
                    optimiser=nn_optimiser,
                    reg = args.reg,
                    loggers=[logger, saver_obj]) # Initialize training class
    
                trainer.train()

                # Evaluate 
                est_filters, estimation_dict = trainer.evaluate()

                saver_obj.save_results(estimation_dict, est_filters)

# Default paths relative to this repository (portable across machines/OSes)
repo_root = Path(__file__).resolve().parent.parent
default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--n_samples", type=int, default=128)
parser.add_argument("--stimulus", type=str, default="random")
parser.add_argument("--sofa_path", type=str, default=str(default_sofa_path))
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--init", type=int, default=0, help="If 1, initialise NN with delta-like impulse responses, if 2 initialise with NN with delta-like impulse response at left speaker and random FIR at right speaker")
parser.add_argument("--reg", type=int, default=2, help="If 1, use first moment or 2 for second moment regularisation during training")
parser.add_argument("--seed", type=int, default = None, help="Random seed for reproducibility")
parser.add_argument("--log", type=int, default=1, help="If 1, will log the training in Tensorboard, to view run in terminal | tensorboard --logdir runs | ")
args = parser.parse_args()

run_battery(args)
            

