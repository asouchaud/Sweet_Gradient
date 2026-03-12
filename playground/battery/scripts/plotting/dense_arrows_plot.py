import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from pathlib import Path
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.setup.setup import load_stim
from scripts.setup.setup import load_hrirs
from scripts.setup.setup import load_gtfb

from scripts.propagation.stereo_anechoic import stereo_anechoic

from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching

DTYPE = torch.float32
repo_root = Path(__file__).resolve().parent
DEFAULT_SOFA_PATH = repo_root / ".." / ".." / ".." / ".."/ "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
DEFAULT_SOFA_IDX = repo_root  / ".." / ".." / ".." / ".."/ "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"


class dense_battery_arrow_plot():
    def __init__(self, folder_path, n_x, n_y, sofa_path =  DEFAULT_SOFA_PATH, sofa_idx = DEFAULT_SOFA_IDX):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fs = 48000
        self.sofa_path = sofa_path
        self.sofa_idx = sofa_idx
        self.curr_loc = None
        self.curr_unc = None
        self.curr_col = None

        self.folder_path = Path(folder_path) 
        self.get_folders()

        json_file_path = os.path.join(self.folder_path, "config.json")
        json_data = json.load(open(json_file_path))
        self.nf = json_data['near_field']
        self.stim_type = json_data['stimulus']

        self.pos_dict = {0 : "1_pos", 1: "3_pos", 2: "5_pos", 3: "9_pos"}
        self.angle_dict = {0: "000_deg", 1: "010_deg", 2: "020_deg", 3: "340_deg", 4: "350_deg"}
        self.opt_dict = {0: "lxx.npy", 1: "xux.npy", 2: "xxc.npy", 3: "lux.npy", 4: "lxc.npy", 5: "xuc.npy", 6: "luc.npy"}

        self.inv_pos_dict = {"1_pos": 0,"3_pos": 1, "5_pos": 2, "9_pos": 3}
        self.inv_angle_dict = {"000_deg": 0, "010_deg": 1, "020_deg": 2, "340_deg": 3, "350_deg": 4}
        self.inv_opt_dict = {"lxx.npy": 0, "xux.npy": 1, "xxc.npy": 2, "lux.npy": 3, "lxc.npy": 4, "xuc.npy": 5, "luc.npy": 6}

        # Init structure to store figures
        self.figs = {}

        self.grid = self.create_grid(n_x, n_y)
        self.spk_pos = torch.tensor([[1,1], [1,-1]], dtype=DTYPE, device=self.device)

        self.load_figs()

    def get_folders(self):

                # POS folders
        self.pos_folder = sorted([
            f.name for f in self.folder_path.iterdir()
            if f.is_dir()
        ])

        self.n_lstn_pos_conf = len(self.pos_folder)


        # ANGLE folders (look inside first pos)
        first_pos_path = self.folder_path / self.pos_folder[0]

        self.angle_folder = sorted([
            f.name for f in first_pos_path.iterdir()
            if f.is_dir()
        ])

        self.n_angles = len(self.angle_folder)


        # NPY files (look inside first angle → np_files)
        first_angle_path = first_pos_path / self.angle_folder[0] / "np_files"

        self.opt_files = sorted([
            f.name for f in first_angle_path.glob("*.npy")
        ])

        self.n_opt = len(self.opt_files)

    def create_grid(self, n_points_x, n_points_y):
        x = np.linspace(-0.75, 0.75, n_points_x)
        y = np.linspace(-0.75, 0.75, n_points_y)

        X, Y = np.meshgrid(x, y, indexing='ij')

        grid = np.stack((X, Y), axis=-1).reshape(-1,2) 

        grid[:,0] -= 0.732 # centered on sweet spot

        return torch.tensor(grid, dtype=DTYPE, device=self.device)
    
    def setup_evaluation(self):

        self.stim = load_stim(self.fs, self.device, DTYPE, self.stim_type)
        self.prop_model = stereo_anechoic(self.stim, 
                                    self.grid, 
                                    spk_pos=self.spk_pos, 
                                    sofa_file_path=self.sofa_path)
        
        self.hrirs = load_hrirs(DTYPE, device=self.device, sofa_path=self.sofa_path, idx_path=self.sofa_idx)
        self.gtfb = load_gtfb(self.fs, device=self.device, dtype=DTYPE)
        self.localisation_model = llado2025(torch.nn.functional.mse_loss, 
                                    0, 
                                    hrirs=self.hrirs,
                                    fs=self.fs,
                                    gtfb=self.gtfb
                                    )
        
        self.uncertainty_model = desena2020(torch.nn.functional.mse_loss,
                                    0,
                                    hrirs=self.hrirs,
                                    fs=self.fs,
                                    gtfb=self.gtfb)
        
        # self.colouration_model = spectral_matching(torch.nn.functional.mse_loss,
        #                                     0,
        #                                     stim=self.stim,
        #                                     hrirs=torch.ones(1, device=self.device, dtype=DTYPE),
        #                                     target_angle=self.target_angle,
        #                                     fs=self.fs
        #                                     )
    
    def get_filters(self, pos_idx, angle_idx, opt_idx):

        self.curr_pos_folder = self.pos_folder[pos_idx]
        self.curr_angle_folder = self.angle_folder[angle_idx]
        self.curr_opt_file = self.opt_files[opt_idx]

        self.filters_path = self.folder_path / self.curr_pos_folder / self.curr_angle_folder / "np_files" / self.curr_opt_file
        self.curr_filters = torch.tensor(np.load(self.filters_path), device = self.device, dtype=DTYPE).unsqueeze(0)

        if self.nf:
            self.target_angle = torch.tensor(int(self.curr_angle_folder.split('_')[0]) % 360, dtype=DTYPE, device=self.device)
        else: 
            self.target_angle = None

    def evaluate_dense_grid(self):

        bin_stim = self.prop_model(self.curr_filters)

        self.curr_loc = self.localisation_model(bin_stim)
        self.curr_unc = self.uncertainty_model(bin_stim)
        # self.curr_col = self.colouration_model(bin_stim)

    def plot_arrows(self):
            
        if self.curr_unc is None:
            unc = torch.zeros_like(self.curr_loc, device=self.device, dtype=DTYPE)
        
        if self.curr_col is None:
            col = torch.zeros_like(self.curr_loc, device=self.device, dtype=DTYPE)
        
        lstn_pos = self.grid.detach().cpu().numpy().squeeze().reshape(-1, 2) # WORKS ONLY FOR BATCH SIZE 1
        spk_pos = self.spk_pos.detach().cpu().numpy().squeeze()
        loc = self.curr_loc.detach().cpu().numpy().squeeze().reshape(-1)
        unc = self.curr_unc.detach().cpu().numpy().squeeze().reshape(-1)
        # col = self.col.detach().cpu().numpy().squeeze().reshape(-1)

        fig_arrow, ax = plt.subplots(figsize=(6,6))
        ax.scatter(spk_pos[0], spk_pos[1], c='red', label='Speaker')
        ax.scatter(lstn_pos[:,0], lstn_pos[:,1], c='blue', label='Listener positions')

        if self.target_angle is not None:
            loc_pos = [1, np.tan(np.deg2rad(self.target_angle.item()))]
            ax.scatter(loc_pos[0], loc_pos[1], c='green', label='Target position')

        
        arrow_len = 0.2
        for x, y, ang, unc in zip(lstn_pos[:,0], lstn_pos[:,1], loc, unc):  

            dx = (1-unc)*arrow_len * np.cos(np.deg2rad(ang))
            dy = (1-unc)*arrow_len * np.sin(np.deg2rad(ang))

            ax.arrow(float(x), float(y), float(dx), float(dy),
                    head_width=0.05, head_length=0.08, fc='orange', ec='orange')    

        ax.set_ylim(-1.25,1.25)
        ax.set_xlim(-0.5,2.0)
        ax.grid()
        ax.set_ylabel('Y position (m)')
        ax.set_xlabel('X position (m)')
        ax.legend()

        title = self.pos_dict[self.inv_pos_dict[self.curr_pos_folder]] + "," + self.angle_dict[self.inv_angle_dict[self.curr_angle_folder]] + "," + self.opt_dict[self.inv_opt_dict[self.curr_opt_file]]
        plt.suptitle(title)
        
        self.figs[(self.curr_pos_folder, self.curr_angle_folder, self.curr_opt_file)] = fig_arrow

        plt.close()

    def load_figs(self):

        # Init models
        self.setup_evaluation()

        # Estimate attributes sequentially
        for i_pos in range(self.n_lstn_pos_conf):
            for i_angle in range(self.n_angles):
                for i_opt in range(self.n_opt):

                    self.get_filters(i_pos, i_angle, i_opt)
                    self.evaluate_dense_grid()
                    self.plot_arrows()

    def return_plot(self, pos_idx, angle_idx, opt_idx, screen, window_size=None):
        # get the figure from your stored figures
        fig = self.figs[(self.pos_folder[pos_idx],
                        self.angle_folder[angle_idx],
                        self.opt_files[opt_idx])]
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # set window_size to screen size if not provided
        if window_size is None:
            window_size = screen.get_size()
        width, height = window_size

        # get ARGB buffer and convert to RGB
        raw_data = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        buf_width, buf_height = canvas.get_width_height()
        raw_data = raw_data.reshape((buf_height, buf_width, 4))
        rgb_data = raw_data[:, :, [1, 2, 3]]  # drop alpha, reorder channels

        # scale or crop to window_size if needed
        surf = pygame.image.frombuffer(rgb_data.tobytes(), (buf_width, buf_height), "RGB")
        surf = pygame.transform.scale(surf, (width, height))  # fit to window_size
        return surf

parser = argparse.ArgumentParser(description='Plot dense arrows')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the run folder')
parser.add_argument('--nx', type = int, default = 5)
parser.add_argument('--ny', type = int, default = 5)
parser.add_argument('--sofa_path', type = str, default= DEFAULT_SOFA_PATH)
parser.add_argument('--sofa_idx', type = str, default= DEFAULT_SOFA_IDX)

args = parser.parse_args()

if args.sofa_path and args.sofa_idx is None:
    parser.error("--sofa_path requires --sofa_idx")

if __name__ == "__main__":
        
    plotter = dense_battery_arrow_plot(args.folder_path, 
                                        n_x = args.nx, 
                                        n_y = args.ny, 
                                        sofa_path=args.sofa_path, 
                                        sofa_idx=args.sofa_idx)
                                                            
    pygame.init()
    window_size = (400, 400)
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

    pos_idx = 0
    angle_idx = 0
    opt_idx = 0

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                
                # Stimulus selection
                if event.key == pygame.K_LEFT:
                    pos_idx = (pos_idx - 1) % plotter.n_lstn_pos_conf
                if event.key == pygame.K_RIGHT:
                    pos_idx = (pos_idx + 1) % plotter.n_lstn_pos_conf
                if event.key == pygame.K_DOWN:
                    angle_idx = (angle_idx - 1) % plotter.n_angles
                if event.key == pygame.K_UP:
                    angle_idx = (angle_idx + 1) % plotter.n_angles
                if pygame.K_0 <= event.key <= pygame.K_6:
                    opt_idx = event.key - pygame.K_0

            surface = plotter.return_plot(pos_idx, angle_idx, opt_idx, screen, window_size=screen.get_size())
            screen.fill((0, 0, 0))
            screen.blit(surface, (0, 0))  # draw the figure
            pygame.display.flip()
 
