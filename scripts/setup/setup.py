## Import Libraries

import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
import numpy as np
import scipy
import sofar as sf
from pathlib import Path
import re

from scripts.helpers.peripheral_processing import estimate_interauralcues

from scripts.datasets.stimulus_dataset import get_stim_loader

from scripts.train.train import ListenerBasedLoss

from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching



def set_seed(seed=44):
    """Set random seeds for reproducibility (PyTorch, NumPy, Python random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA (if using GPU)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Make operations deterministic (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_eval_objects(env, 
                       eval_lstn_pos = torch.tensor([[0.0 , 0.0], [0.0, 0.5], [0.0, -0.5]], device='cpu', dtype=torch.float64),
                       n_eval_stim = 10):
    
    eval_stimulus_data_loader, eval_stimulus_dataset = get_stim_loader(env.fs, n_eval_stim, n_eval_stim) # Get new stimulus dataloader for evaluation (to avoid overfitting to training stimuli)

    # Initiate perceptual models
    eval_loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=eval_lstn_pos, spk_pos=env.spk_pos, device=env.device, dtype=env.dtype) # Llado et al. 2025 model
    eval_col_model = spectral_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=eval_lstn_pos, device=env.device, dtype=env.dtype) # Spectral Matching model
    eval_unc_model = desena2020(loss = F.l1_loss, target = 0, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=eval_lstn_pos, device=env.device, dtype=env.dtype) # De Sena et al. 2020 model

    eval_perceptual_dict = {
            "localisation": ListenerBasedLoss(model=eval_loc_model, weight=1.0, device=env.device, dtype=env.dtype),
            "uncertainty": ListenerBasedLoss(model=eval_unc_model, weight=1, device=env.device, dtype=env.dtype),
            "colouration": ListenerBasedLoss(model=eval_col_model, weight=1, device=env.device, dtype=env.dtype)
        }
    
    return eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset


def load_stim(fs, device, dtype, selected_stim = 'random'):
    if selected_stim == 'random':
        stim = torch.randn(1,1,int(0.01*fs), device=device, dtype=dtype)
    elif selected_stim == 'impulse':
        stim = torch.zeros(1,1,int(0.01*fs), device=device, dtype=dtype)
        stim[0,0,0] = 1.0
    elif selected_stim == 'sine':
        stim = torch.sin(2*np.pi*440*torch.arange(0, int(0.01*fs), device=device, dtype=dtype)/fs).view(1,1,-1)
    elif selected_stim == 'male_speech':
        stim_48 = scipy.io.wavfile.read('./data/stimulus/EBU_MaleSpeech.wav')
        stim = torch.tensor(scipy.signal.resample(stim_48[1], int(len(stim_48[1])*fs/stim_48[0])), device=device, dtype=dtype).view(1,1,-1)
        # stim = stim[0,0,int(fs*0.5):fs].view(1,1,-1)
    elif selected_stim == 'pop_music':
        stim_48 = scipy.io.wavfile.read('./data/stimulus/EBU_PopMusic.wav')
        stim = torch.tensor(scipy.signal.resample(stim_48[1], int(len(stim_48[1])*fs/stim_48[0])), device=device, dtype=dtype).view(1,1,-1)
        # stim = stim[0,0,int(fs*0.5):fs].view(1,1,-1)
    else:
        print("Selected stimulus not available, using random noise")
        stim = torch.randn(1,1,int(0.01*fs), device=device, dtype=dtype)
    return stim

def load_gtfb(fs, device, dtype):
    
    # Load Gammatone Filter Bank Impulse Responses
    script_dir = os.path.dirname(__file__)
    gtfb_path = os.path.join(script_dir, "..","..","data", "impulses","GTFB", "GTFB_4096_fs_48000.mat")
    gtfb_path = os.path.normpath(gtfb_path)

    MATLAB_GTFB_dict = scipy.io.loadmat(gtfb_path)
    gtfb = torch.tensor(np.real(MATLAB_GTFB_dict['gtfb'].T), dtype=dtype, device=device).unsqueeze(1)

    return gtfb

def load_hrirs(dtype, device, sofa_path = os.path.join('data', 'impulses', 'SOFA', 'D1_48K_24bit_256tap_FIR_SOFA.sofa'), idx_path = os.path.join( 'data','impulses', 'SOFA', 'D1_48k_hrirs_idx.csv')):
    # Load SOFA File 
    sofaile = sf.read_sofa(sofa_path) #CHANGE HERE TO CHANGE HRIR SET
    idx = np.loadtxt(idx_path, delimiter=",") # FROM RIGHT TO LEFT EAR, 37 AZIMUTHS FROM -90 TO 90 IN STEPS OF 5 DEGREES, 1 ELEVATION (0 DEGREES)

    # Get IRs from SOFA File and Pad for Desired Output 
    hrirs = sofaile.Data_IR[idx.astype(int),:,:]
    hrirs = F.pad(torch.tensor(hrirs, dtype=dtype, device=device),(0,4800))

    return hrirs

def load_template(hrirs, gtfb, fs, device, dtype):
    # Compute Interaural Cues for Template
    template_dict = estimate_interauralcues(hrirs.unsqueeze(0),fs, gtfb=gtfb, tau_IACC = 1e-2)
    template_dict['azimuths'] = torch.arange(-90,95,5,dtype=dtype, device=device, requires_grad=True).view(-1,1)
    template_dict['ild'] = template_dict['ild'].squeeze()
    template_dict['itd'] = template_dict['itd'].squeeze()

    return template_dict

def load_sofa_gtfb_template(device, dtype, fs):
    # Load SOFA File 
    sofa_file_path = './data/SOFA/D1_48K_24bit_256tap_FIR_SOFA.sofa'
    pp55_hrirs = sf.read_sofa(os.path.join('data', 'SOFA', 'D1_48K_24bit_256tap_FIR_SOFA.sofa'))
    pp55_idx = np.loadtxt(os.path.join( 'data', 'SOFA', 'D1_48k_hrirs_idx.csv'), delimiter=",")

    # Get IRs from SOFA File and Pad for Desired Output 
    bin_stim = pp55_hrirs.Data_IR[pp55_idx.astype(int),:,:]
    bin_stim = F.pad(torch.tensor(bin_stim, dtype=dtype, device=device),(0,4800))  # HAD TO ADD FRONT PADDING OF SIZE 1/2 OF FILTER SIZE AND BACK PADDING OF SIZE MATLAB_PADDING - 1/2 OF FILTER SIZE + 1

    ## Load Gammatone Filter Bank Impulse Responses
    script_dir = os.path.dirname(__file__)
    gtfb_path = os.path.join(script_dir, "..","..","..","data", "stimulus", "GTFB_4096_fs_48000.mat")
    gtfb_path = os.path.normpath(gtfb_path)

    MATLAB_GTFB_dict = scipy.io.loadmat(gtfb_path)
    gtfb = torch.tensor(np.real(MATLAB_GTFB_dict['gtfb'].T), dtype=torch.float, device=device).unsqueeze(1)

    # Compute Interaural Cues for Template
    template_dict = estimate_interauralcues(bin_stim.unsqueeze(0),fs, gtfb=gtfb, tau_IACC = 1e-2)
    template_dict['azimuths'] = torch.arange(-90,95,5,dtype=dtype, device=device, requires_grad=True).view(-1,1)
    template_dict['ild'] = template_dict['ild'].squeeze()
    template_dict['itd'] = template_dict['itd'].squeeze()

    return sofa_file_path, gtfb, template_dict


def setup_folder_structure():

    # Directory where your reports and folders live
    base_dir = Path("./data/run_log")

    max_y = 1

    # Regex to match files like "report_3", "report_10.txt", etc.
    pattern = re.compile(r"^report_(\d+)(?:\..*)?$")

    for entry in base_dir.iterdir():
        if entry.is_file():
            match = pattern.match(entry.name)
            if match:
                y = int(match.group(1))
                if y > max_y:
                    max_y = y

    # x = highest y + 1
    x = max_y + 1
    new_folder = Path("./data/estimated/estimated_" + str(x))

    # Create the new folder
    new_folder.mkdir(exist_ok=False)

    print(f"Created folder: {new_folder}")


    # Create folders to save data if they do not exist
    if not os.path.exists(f'./data/estimated/estimated_{x}/filters_reg/'):
        os.makedirs(f'./data/estimated/estimated_{x}/filters_reg/')
    if not os.path.exists(f'./data/estimated/estimated_{x}/IRs_reg/'):
        os.makedirs(f'./data/estimated/estimated_{x}/IRs_reg/')
    if not os.path.exists(f'./data/estimated/estimated_{x}/stim_reg/'):
        os.makedirs(f'./data/estimated/estimated_{x}/stim_reg/')
    if not os.path.exists(f'./data/estimated/estimated_{x}/ref_reg/'):
        os.makedirs(f'./data/estimated/estimated_{x}/ref_reg/')
    if not os.path.exists('./data/run_log/'):
        os.makedirs('./data/run_log/')