import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from pathlib import Path

from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs
from scripts.setup.setup import load_gtfb

from scripts.propagation.stereo_anechoic import stereo_anechoic

from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

DTYPE = torch.float32

def plot_arrows(lstn_pos, spk_pos, loc, unc = None, col=None, loc_target = None, show = True):
        
        if unc is None:
            unc = torch.zeros_like(loc, device=loc.device, dtype=loc.dtype)
        
        if col is None:
            col = torch.zeros_like(loc, device=loc.device, dtype=loc.dtype)
    
        
        lstn_pos = lstn_pos.detach().cpu().numpy().squeeze().reshape(-1, 2) # WORKS ONLY FOR BATCH SIZE 1
        spk_pos = spk_pos.detach().cpu().numpy().squeeze()
        loc = loc.detach().cpu().numpy().squeeze().reshape(-1)
        unc = unc.detach().cpu().numpy().squeeze().reshape(-1)
        col = col.detach().cpu().numpy().squeeze().reshape(-1)

        fig_arrow, ax = plt.subplots(figsize=(6,6))
        ax.scatter(spk_pos[0], spk_pos[1], c='red', label='Speaker')
        ax.scatter(lstn_pos[:,0], lstn_pos[:,1], c='blue', label='Listener positions')

        if loc_target is not None:
            loc_pos = [1, np.tan(np.deg2rad(loc_target.item()))]
            ax.scatter(loc_pos[0], loc_pos[1], c='green', label='Target position')

        
        arrow_len = 0.2
        for x, y, ang, unc in zip(lstn_pos[:,0], lstn_pos[:,1], loc, unc):  

            dx = (1-unc)*arrow_len * np.cos(np.deg2rad(ang))
            dy = (1-unc)*arrow_len * np.sin(np.deg2rad(ang))

            ax.arrow(float(x), float(y), float(dx), float(dy),
                    head_width=0.05, head_length=0.08, fc='orange', ec='orange')    

        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5-0.732,1.5)
        ax.grid()
        ax.set_ylabel('Y position (m)')
        ax.set_xlabel('X position (m)')
        ax.legend()

        if show:
            plt.show()
        
        return fig_arrow


def load_data(folder_path):
    config = json.load(open(folder_path + '/config.json', 'r')) 
    results = json.load(open(folder_path + '/results.json', 'r'))

    optim_lstn_pos = config['listeners_position']
    spk_pos = config['speakers_position']

    stim_type = config['stimulus']
    fs = config['fs']
    target_angle = config['target_angle']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filters_path = os.path.join(folder_path, "artifacts", "est_filters.npy")
    filters = torch.tensor(np.load(filters_path), dtype=DTYPE, device=device).unsqueeze(0)

    return filters, optim_lstn_pos, spk_pos, target_angle, stim_type, fs, device

def create_grid(n_points_x, n_points_y):
    x = np.linspace(-1, 1, n_points_x)
    y = np.linspace(-1, 1, n_points_y)

    X, Y = np.meshgrid(x, y, indexing='ij')

    grid = np.stack((X, Y), axis=-1).reshape(-1,2) 

    grid[:,0] -= 0.732 # centered on sweet spot

    return grid

def setup_evaluation(dense_grid, spk_pos, target_angle, stim_type, fs, device, args):

    stim = load_stim(fs, device, DTYPE, stim_type)
    prop_model = stereo_anechoic(stim, 
                                  dense_grid, 
                                  spk_pos=spk_pos, 
                                  sofa_file_path=args.sofa_path)
    
    hrirs = load_hrirs(DTYPE, device=device, sofa_path=args.sofa_path, idx_path=args.sofa_idx)
    gtfb = load_gtfb(fs, device=device, dtype=DTYPE)
    localisation_model = llado2025(torch.nn.functional.mse_loss, 
                                   0, 
                                   hrirs=hrirs,
                                   fs=fs,
                                   gtfb=gtfb
                                   )
    
    uncertainty_model = desena2020(torch.nn.functional.mse_loss,
                                   0,
                                   hrirs=hrirs,
                                   fs=fs,
                                   gtfb=gtfb)
    
    colouration_model = spectral_matching(torch.nn.functional.mse_loss,
                                          0,
                                          stim=stim,
                                          hrirs=torch.ones(1, device=device, dtype=DTYPE),
                                          target_angle=target_angle,
                                          fs=fs
                                          )
    
    return prop_model, localisation_model, uncertainty_model, colouration_model


def evaluate_dense_grid(filters, prop_model, localisation_model, uncertainty_model, colouration_model):

    bin_stim = prop_model(filters)

    loc = localisation_model(bin_stim)
    unc = uncertainty_model(bin_stim)
    col = colouration_model(bin_stim)

    return loc, unc, col


repo_root = Path(__file__).resolve().parent
default_sofa_path = repo_root / ".." / ".."/ "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
default_sofa_idx = repo_root  / ".." / ".."/ "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"

parser = argparse.ArgumentParser(description='Plot dense arrows')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the run folder')
parser.add_argument('--sofa_path', type = str, default= default_sofa_path)
parser.add_argument('--sofa_idx', type = str, default= default_sofa_idx)

args = parser.parse_args()

if args.sofa_path and args.sofa_idx is None:
    parser.error("--sofa_path requires --sofa_idx")

if __name__ == "__main__":
    
    filters, opt_lstn_pos, spk_pos, target_angle, stim_type, fs, device = load_data(args.folder_path)

    dense_grid = create_grid(10,10)

    prop_model, localisation_model, uncertainty_model, colouration_model = setup_evaluation(dense_grid,
                                                                                            spk_pos, 
                                                                                            target_angle, 
                                                                                            stim_type,
                                                                                            fs,
                                                                                            device,
                                                                                            args
                                                                                            )
    
    loc, unc, col = evaluate_dense_grid(filters, prop_model, localisation_model, uncertainty_model, colouration_model)
    
    plot_arrows(torch.tensor(dense_grid),torch.tensor(spk_pos),loc,unc,col,torch.tensor(target_angle),True)


    
