# Import computational libraries
import torch
import torch.nn.functional as F

# Import libraries for plotting
from scripts.plotting.graph_plot import graph_plot

# Import setup functions
from scripts.setup.setup import set_seed
from scripts.setup.setup import load_gtfb
from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs

# Import iteration library
from itertools import combinations


def setup(args):
    # Set data type, device and sampling frequency
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = 48000

    # Set random seed for reproducibility
    set_seed(seed=args.seed)

    # Training angles 
    target_angle = torch.tensor([0,10,-10, 20, -20], device=device, dtype=dtype)  # Target angle in degrees

    # Loudspeaker and listener positions
    lstn_pos_conf = [[[-0.732, 0]],  # 1 listener position
                [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5]], # 3 listener positions
                [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5], [-0.232, 0], [-1.232, 0], [-0.232, 0.5], [-0.232, -0.5], [-1.232, 0.5], [-1.232, -0.5]]] # 9 listener positions
    spk_pos = [[1,1], [1,-1]]

## Load data

    gtfb = load_gtfb(fs=fs, device=device, dtype=dtype)  # Load Gamma Tone Filter Bank
    stim = load_stim(fs=fs, device=device, dtype=dtype, selected_stim=args.stimulus)  # Load stimulus
    hrirs = load_hrirs(dtype=dtype, device=device) # Load HRIRs using specific location from SOFA file

    return dtype, device, fs, target_angle, lstn_pos_conf, spk_pos, gtfb, stim, hrirs


def get_all_subsets(classes):

    dict_subsets = [
    {k: v for d in combo for k, v in d.items()}
    for r in range(1, len(classes) + 1)
    for combo in combinations(classes, r)
    ]
    return dict_subsets

def init_perc_model(full_dict, device, dtype, fs, stim, hrirs, target_angle, gtfb):
    perceptual_dict = {}
    for key, model_class in full_dict.items():
        if key == 'localisation':
            model_instance = model_class(F.mse_loss, target=target_angle, hrirs=hrirs, fs=fs, gtfb=gtfb) # Initialize model instance
        elif key == 'uncertainty':
            model_instance = model_class(F.mse_loss, torch.tensor([0], device = device, dtype=dtype), hrirs=hrirs, fs=fs, gtfb=gtfb) # Initialize model instance
        elif key == 'colouration':
            model_instance = model_class(F.l1_loss, torch.tensor([0], device = device, dtype=dtype), stim = stim, hrirs=hrirs, target_angle = target_angle, fs=fs) # Initialize model instance
        else:
            Exception(f"Perceptual model {key} not recognized.")
        perceptual_dict[key] = model_instance # Store in dictionary
    return perceptual_dict


