import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors
import argparse
import json
import os
from pathlib import Path

from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs
from scripts.setup.setup import load_gtfb

from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation

from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

from scripts.perceptual_models.llado2025.llado2025_helpers import near_field_comp_angles

DTYPE = torch.float32

def plot_arrows(lstn_pos, spk_pos, loc, unc = None, col=None, loc_target = None, show = True, vbap_values = None):
        
        unc_was_none = unc is None
        col_was_none = col is None
        
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
        ax.scatter(spk_pos[:,0], spk_pos[:,1], c='red', label='Speaker')
        ax.scatter(lstn_pos[:,0], lstn_pos[:,1], c='blue', label='Listener positions')
        spk_x_mid = 0.5 * (float(np.min(spk_pos[:, 0])) + float(np.max(spk_pos[:, 0])))
        for source_idx, (sx, sy) in enumerate(spk_pos):
            # Small labels to avoid overlap in tightly spaced arrays.
            place_right = float(sx) < spk_x_mid
            dx = 0.022 if place_right else -0.022
            dy = 0.010 if (source_idx % 2 == 0) else -0.010
            ha = "left" if place_right else "right"
            ax.text(
                float(sx) + dx,
                float(sy) + dy,
                str(source_idx),
                color="white",
                fontsize=6,
                fontweight="bold",
                ha=ha,
                va="center",
                bbox=dict(boxstyle="round,pad=0.10", fc="black", ec="none", alpha=0.80),
                zorder=8,
                clip_on=True,
            )
        
        target_angle = None
        if loc_target is not None:
            loc_pos = [1.732, 1.732*np.tan(np.deg2rad(loc_target.item()))]
            ax.scatter(loc_pos[0], loc_pos[1], c='green', label='Target position')

            target_angle = near_field_comp_angles(torch.tensor(loc_pos), lstn_pos)
            target_angle = target_angle.detach().cpu().numpy().squeeze().reshape(-1)
        
        arrow_len = 0.2

            # Create a colormap with log scaling
        max_col = 18.5
        norm = colors.Normalize(vmin=0, vmax=max_col)  # avoid zero
        cmap = cm.viridis
        for x, y, ang, leng, col_val in zip(lstn_pos[:,0], lstn_pos[:,1], loc, unc, col):  

            dx = (1-leng)*arrow_len * np.cos(np.deg2rad(ang))
            dy = (1-leng)*arrow_len * np.sin(np.deg2rad(ang))

            color = cmap(norm(max(col_val, 1e-3)))  # avoid zero for log scale

            ax.arrow(float(x), float(y), float(dx), float(dy),
                    head_width=0.05, head_length=0.08, fc=color, ec=color)    

            # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig_arrow.colorbar(sm, ax=ax)
        cbar.set_label("LSD")


        if vbap_values is not None:

            loc_vbap, unc_vbap, col_vbap = vbap_values
            
            loc_vbap = loc_vbap.detach().cpu().numpy().squeeze().reshape(-1)
            unc_vbap = unc_vbap.detach().cpu().numpy().squeeze().reshape(-1)
            col_vbap = col_vbap.detach().cpu().numpy().squeeze().reshape(-1)

            arrow_len = 0.2
            for x, y, ang, leng in zip(lstn_pos[:,0], lstn_pos[:,1], loc_vbap, unc_vbap):  

                dx = (1-leng)*arrow_len * np.cos(np.deg2rad(ang))
                dy = (1-leng)*arrow_len * np.sin(np.deg2rad(ang))

                ax.arrow(float(x), float(y), float(dx), float(dy),
                        head_width=0.05, head_length=0.08, fc='purple', ec='purple')
                

        legend_elements = [
            Line2D([0], [0], color='orange', lw=2, label='Model prediction'),
        ]

        if vbap_values is not None:
            legend_elements.append(
                Line2D([0], [0], color='purple', lw=2, label='VBAP prediction')
            )

        ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_elements,
                labels=ax.get_legend_handles_labels()[1] + [le.get_label() for le in legend_elements])

        ax.set_ylim(-1.25,1.25)
        ax.set_xlim(-0.5,2.0)
        ax.grid()
        ax.set_ylabel('Y position (m)')
        ax.set_xlabel('X position (m)')

        # --- format caption text ---
        parts = [f"loc: {np.array2string(loc, precision=2, separator=', ')}"]

        if target_angle is not None:
            parts.append(f"target: {np.array2string(target_angle, precision=2, separator=', ')}")

        if not unc_was_none:
            parts.append(f"unc: {np.array2string(unc, precision=2, separator=', ')}")

        if not col_was_none:
            parts.append(f"col: {np.array2string(col, precision=2, separator=', ')}")

        caption = " | ".join(parts)

        # make space for caption
        fig_arrow.subplots_adjust(bottom=0.22)

        # add caption
        fig_arrow.text(
            0.5, 0.02, caption,
            ha='center',
            va='bottom',
            fontsize=9,
            wrap=True
        )


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
    x = np.linspace(-0.75, 0.75, n_points_x)
    y = np.linspace(-0.75, 0.75, n_points_y)

    X, Y = np.meshgrid(x, y, indexing='ij')

    grid = np.stack((X, Y), axis=-1).reshape(-1,2) 

    return grid

def setup_evaluation(dense_grid, spk_pos, target_angle, stim_type, fs, device, args):

    stim = load_stim(fs, device, DTYPE, stim_type)
    prop_model = BinauralFreeFieldPropagation(stim, 
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


def azimuth_to_unit_vector(azimuth_deg):
    rad = np.deg2rad(azimuth_deg)
    return np.array([np.cos(rad), np.sin(rad)])

def vbap_stereo_gains(source_azimuth_deg, ls_azimuths_deg=(-30, 30)):
    l1 = azimuth_to_unit_vector(ls_azimuths_deg[0])
    l2 = azimuth_to_unit_vector(ls_azimuths_deg[1])
    L = np.column_stack((l1, l2))
    p = azimuth_to_unit_vector(source_azimuth_deg)

    try:
        g = np.linalg.solve(L, p)
    except np.linalg.LinAlgError:
        raise ValueError("Invalid speaker configuration.")

    if np.any(g < 0):
        g[:] = 0.0
    else:
        g = g / np.linalg.norm(g)

    return g

repo_root = Path(__file__).resolve().parent
default_sofa_path = repo_root / ".." / ".."/ "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
default_sofa_idx = repo_root  / ".." / ".."/ "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"
vbap_values = None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot dense arrows')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the run folder')
    parser.add_argument('--sofa_path', type = str, default= default_sofa_path)
    parser.add_argument('--sofa_idx', type = str, default= default_sofa_idx)
    parser.add_argument("--vbap_comp", type = int, default=1, help = "Display VBAP localisation for desired angle")

    args = parser.parse_args()

    if args.sofa_path and args.sofa_idx is None:
        parser.error("--sofa_path requires --sofa_idx")
    
    filters, opt_lstn_pos, spk_pos, target_angle, stim_type, fs, device = load_data(args.folder_path)

    dense_grid = create_grid(3,3)


    prop_model, localisation_model, uncertainty_model, colouration_model = setup_evaluation(dense_grid,
                                                                                            spk_pos, 
                                                                                            target_angle, 
                                                                                            stim_type,
                                                                                            fs,
                                                                                            device,
                                                                                            args
                                                                                            )

    if args.vbap_comp:
        g = vbap_stereo_gains(target_angle)

        filters_vbap = torch.zeros((1,2,256),dtype=DTYPE, device=device)
        filters_vbap[0,:,0] = torch.tensor(g, dtype=DTYPE, device=device)

        vbap_values = evaluate_dense_grid(filters_vbap, prop_model, localisation_model, uncertainty_model, colouration_model)                                                                                      
    
    loc, unc, col = evaluate_dense_grid(filters, prop_model, localisation_model, uncertainty_model, colouration_model)


    
    plot_arrows(torch.tensor(dense_grid),torch.tensor(spk_pos),loc,unc,col,torch.tensor(target_angle),True, vbap_values = vbap_values)


    
