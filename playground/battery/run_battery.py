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
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation

# Import perceptual models
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

# Import neural network related scripts
from scripts.optimisation.nn_init import load_NN
from scripts.optimisation.nn_init import load_optimiser
from scripts.optimisation.nn_init import load_scheduler
from scripts.optimisation.regularisation import first_moment_regularisation
from scripts.optimisation.regularisation import second_moment_regularisation

# Import training script
from scripts.train.train import Train

# Import saving functions
from playground.battery.scripts.saving.save_results import saver

# Import time measure
from datetime import datetime

# Import plotting functions
from scripts.plotting.dense_arrows_plot import plot_arrows
from scripts.plotting.graph_plot import plot_time
from scripts.plotting.graph_plot import plot_freq
from scripts.plotting.graph_plot import plot_color

def build_perceptual_loss_term(model, device, dtype):
    def loss_term(bin_stim, h_full):  # noqa: ARG001 - kept for a uniform callable interface
        estimation = model(bin_stim)
        target = torch.as_tensor(model.target, device=device, dtype=dtype).expand_as(estimation)
        loss = model.loss(estimation, target) / model.loss_norm
        return loss, estimation

    return loss_term

def build_regularisation_loss_term(regularisation, source_idx):
    def loss_term(bin_stim, h_full):  # noqa: ARG001 - kept for a uniform callable interface
        source_filter = h_full[:, source_idx:source_idx + 1, :]
        loss = regularisation.compute_loss(source_filter) / regularisation.loss_norm
        return loss, None

    return loss_term

def select_regularistion(reg: str, n_samples: int, device='cuda', dtype=torch.float32):
    if reg == 0:
        return None
    if reg == 1:
        return first_moment_regularisation(n_samples=n_samples, device=device, dtype=dtype)
    if reg == 2:
        return second_moment_regularisation(n_samples=n_samples, device=device, dtype=dtype)
    return None


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

    ## QUICK FIX TO GO FASTER
    dict_idx = [0,3,4,6]
    
    for lstn_pos_conf_idx in range(n_lstn_pos_conf):
        for ta_idx in range(n_ta):
            for dict_sub_idx in range(4): # 7 different initialisation and regularisation combinations  ## CHANGE THIS BACK TO DICT_SUB_IDX
                print(f"Running battery for target angle {target_angle[ta_idx].item()} deg, listener position configuration {lstn_pos_conf_idx}, current run {dict_idx[dict_sub_idx]+1}/7") ## CHANGE THIS BACK TO DICT_SUB_IDX
                
                # Initialize sound propagation module
                stereo_prop = BinauralFreeFieldPropagation(stim=stim,
                                 lstn_pos=lstn_pos_conf[lstn_pos_conf_idx],
                                 spk_pos=spk_pos,
                                 fs=fs,
                                 sofa_file_path=default_sofa_path) # Load stereo anechoic propagation module
                
                # Initialize model
                if args.near_field:
                    perceptual_dict = init_perc_model(dict_subsets[dict_idx[dict_sub_idx]], device, dtype, fs, stim, hrirs, target_angle[ta_idx], gtfb, lstn_pos_conf[lstn_pos_conf_idx]) ## CHANGE THIS BACK TO DICT_SUB_IDX
                else:
                    perceptual_dict = init_perc_model(dict_subsets[dict_idx[dict_sub_idx]], device, dtype, fs, stim, hrirs, target_angle[ta_idx], gtfb) ## CHANGE THIS BACK TO DICT_SUB_IDX
                loss_terms = {
                    key: build_perceptual_loss_term(model, device=device, dtype=dtype)
                    for key, model in perceptual_dict.items()
                }
                regularisation = select_regularistion(args.reg, n_samples=args.n_samples, device=device, dtype=dtype)
                if regularisation is not None:
                    for source_idx in range(len(spk_pos)):
                        loss_terms[f"regularisation_source_{source_idx}"] = build_regularisation_loss_term(
                            regularisation,
                            source_idx=source_idx,
                        )
                
                # Initialise neural network
                nn_model = load_NN(n_samples=args.n_samples, device=device, dtype=dtype, init = args.init) # Load neural network ICLD or FIR
                nn_optimiser = load_optimiser(nn_model, lr=0.01) # Load optimiser for neural network
                # nn_scheduler = load_scheduler(nn_optimiser,args.epochs)
                nn_scheduler = None

                saver_obj = saver(current_date=current_date,
                                  current_time=current_time,
                                  reg_idx=args.reg, 
                                  init_idx=args.init, 
                                  n_samples=args.n_samples, 
                                  nf_idx=args.near_field,
                                  angle=target_angle[ta_idx], 
                                  lstn_len=len(lstn_pos_conf[lstn_pos_conf_idx]),
                                  curr_idx=dict_idx[dict_sub_idx]) ## CHANGE THIS BACK TO DICT_SUB_IDX
                logger = saver_obj if args.log else None

                # Train the neural network
                trainer = Train(n_epoch=args.epochs,
                    nn=nn_model,
                    prop=stereo_prop,
                    perceptual_dict=perceptual_dict,
                    loss_terms=loss_terms,
                    optimiser=nn_optimiser,
                    loggers=[logger, saver_obj],
                    scheduler=nn_scheduler) # Initialize training class
    
                trainer.train()

                # Evaluate 
                if args.near_field:
                    full_perceptual_dict = init_perc_model(dict_subsets[-1], device, dtype, fs, stim, hrirs, target_angle[ta_idx], gtfb, lstn_pos_conf[lstn_pos_conf_idx]) ## CHANGE THIS BACK TO DICT_SUB_IDX
                else:
                    full_perceptual_dict = init_perc_model(dict_subsets[-1], device, dtype, fs, stim, hrirs, target_angle[ta_idx], gtfb) ## CHANGE THIS BACK TO DICT_SUB_IDX
                
                est_filters, estimation_dict, bin_stim = trainer.evaluate(full_perceptual_dict)

                saver_obj.save_results(estimation_dict, est_filters)

                with torch.no_grad():
                    time_plot = plot_time(est_filters, show=False)
                    freq_plot = plot_freq(est_filters, fs, show=False)
                    arrow_plot = None
                    if 'localisation' in full_perceptual_dict:
                        if full_perceptual_dict['localisation'].lstn_pos_conf is not None:
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
                        color_l_plot, color_r_plot = plot_color(bin_stim, full_perceptual_dict['colouration'].reference_fft, stereo_prop.lstn_pos, show=False)
                    
                    saver_obj.save_plots(time_plot,freq_plot,arrow_plot,color_l_plot,color_r_plot)
    
    saver_obj.save_config(nf=args.near_field,
                          epochs=args.epochs,
                          n_samples=args.n_samples,
                          stimulus=args.stimulus,
                          sofa_path=args.sofa_path,
                          lr=args.lr,
                          init=args.init,
                          reg=args.reg,
                          seed=args.seed,
                          log=args.log)

# Default paths relative to this repository (portable across machines/OSes)
repo_root = Path(__file__).resolve().parent.parent
default_sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"

parser = argparse.ArgumentParser()
parser.add_argument("--near_field", type = int, default = 0, help = "Optimise for source in the near-field (0) and far-field (1)")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--n_samples", type=int, default=128)
parser.add_argument("--stimulus", type=str, default="random")
parser.add_argument("--sofa_path", type=str, default=str(default_sofa_path))
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--init", type=int, default=1, help="If 1, initialise NN with delta-like impulse responses, if 2 initialise with NN with delta-like impulse response at left speaker and random FIR at right speaker")
parser.add_argument("--reg", type=int, default=1, help="If 1, use first moment or 2 for second moment regularisation during training")
parser.add_argument("--alpha", type=float, default=1.5)
parser.add_argument("--seed", type=int, default=52, help="Random seed for reproducibility")
parser.add_argument("--log", type=int, default=1, help="If 1, will log the training in Tensorboard, to view run in terminal | tensorboard --logdir runs | ")
args = parser.parse_args()

run_battery(args)
            

