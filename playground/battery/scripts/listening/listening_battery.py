import numpy as np
import scipy
import re
import sounddevice as sd
import torch
import os 
import json
import pygame
import argparse

from scripts.setup.setup import load_stim

from scripts.propagation.stereo_anechoic import stereo_anechoic


class binaural_listening:
    def __init__(self, folder_path):

        self.folder_path = folder_path 

        # self.json_file_path = os.path.join(self.folder_path, "config.json")
        # json_data = json.load(open(self.json_file_path))

        self.pos_dict = {0 : "1_pos", 1: "3_pos", 2: "5_pos", 3: "9_pos"}
        self.angle_dict = {0: "000_deg", 1: "010_deg", 2: "020_deg", 3: "340_deg", 4: "350_deg"}
        self.opt_dict = {0: "lxx.npy", 1: "xux.npy", 2: "xxc.npy", 3: "lux.npy", 4: "lxc.npy", 5: "xuc.npy", 6: "luc.npy"}

        self.inv_pos_dict = {"1_pos": 0,"3_pos": 1, "5_pos": 2, "9_pos": 3}
        self.inv_angle_dict = {"000_deg": 0, "010_deg": 1, "020_deg": 2, "340_deg": 3, "350_deg": 4}
        self.inv_opt_dict = { "lxx.npy": 0, "xux.npy": 1, "xxc.npy": 2, "lux.npy": 3, "lxc.npy": 4, "xuc.npy": 5, "luc.npy": 6}

        
        # Loudspkeaker and listener positions
        self.fs = 48000
        self.spk_pos = [[1,1], [1,-1]]
        self.lstn_pos_conf = [[[-0.732, 0.0]],  # 1 listner position
                    [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5]], # 3 listener positions
                    [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5], [-0.232, 0], [-1.232, 0]], # 5 listener positions
                    [[-0.732, 0], [-0.732, 0.5], [-0.732, -0.5], [-0.232, 0], [-1.232, 0], [-0.232, 0.5], [-0.232, -0.5], [-1.232, 0.5], [-1.232, -0.5]]] # 9

        # self.sofa_path = json_data["sofa_path"]
        self.sofa_path = r"data\impulses\SOFA\D1_48K_24bit_256tap_FIR_SOFA.sofa"
        
        self.stim = load_stim(self.fs, device='cuda', dtype=torch.float32, selected_stim="random")

        self.filter_path = os.path.join(folder_path, self.pos_dict[0], self.angle_dict[0],"np_files", self.opt_dict[0])

        if not os.path.exists(self.filter_path):
            print(f"Warning: Filter not found at {self.filter_path}, searching for fallback...")
            fallback_path = self.find_any_filter(self.folder_path)
            if fallback_path is None:
                raise FileNotFoundError(f"No .npy filters found under {self.folder_path}")
            print(f"Using fallback filter: {fallback_path}")
            self.filter_path = fallback_path

        # Split path into components
        parts = self.filter_path.replace("\\", "/").split("/")  # works for Windows

        # Extract pos, angle, and opt names
        pos_name = parts[-4]    # '3_pos'
        angle_name = parts[-3]  # '020_deg'
        opt_name = parts[-1]    # 'xux.npy'

        self.pos_conf_idx = self.inv_pos_dict[pos_name]
        self.pos_idx = 0
        self.angle_idx = self.inv_angle_dict[angle_name]
        self.stim_idx = 0
        self.opt_idx = self.inv_opt_dict[opt_name]


        self.prop = stereo_anechoic(self.stim, self.lstn_pos_conf[self.pos_conf_idx], self.spk_pos, self.sofa_path, self.fs)

        filters_np = np.load(self.filter_path)
        self.filters = torch.tensor(filters_np, device='cuda', dtype=torch.float32).unsqueeze(0)
        
        self.bin_stim = self.prop(self.filters)


    def find_any_filter(self, folder_path):
        # Walk through all subfolders
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.npy'):
                    return os.path.join(root, f)
        return None

    def listen(self):

        # Convert to numpy and play
        bin_stim_np = self.bin_stim.cpu().detach().numpy().squeeze()
        sd.play(bin_stim_np[self.pos_idx].T, self.fs)
        sd.wait()

    def select_lstn_conf(self, pos_conf_idx):
        self.pos_conf_idx = pos_conf_idx
        print(f"Selected listener position configuration: {self.pos_dict[self.pos_conf_idx]}")
    
    def select_stim(self, stim_idx):

        stim_dict = {0: "random", 1: "sine", 2: "male_speech", 3: "pop_music"}
        self.stim = load_stim(self.fs, device='cuda', dtype=torch.float32, selected_stim=stim_dict[stim_idx])
        print(f"Selected stimulus: {stim_dict[stim_idx]}")

    def select_optimisation(self, opt_idx):

        if opt_idx > 6 or opt_idx < 0:
            print("Invalid optimisation index. Please select a value between 0 and 6. Defaulting to localisation.")
            opt_idx = 0
        self.opt_idx = opt_idx
        print(f"Selected optimisation target: {self.opt_dict[opt_idx]}")

    def load_filters(self):
        try:
            # Save old filters in case of error
            old_filters = self.filters

            # Build filter path
            filter_path = os.path.join(
                self.folder_path,
                self.pos_dict[self.pos_conf_idx],
                self.angle_dict[self.angle_idx],
                "np_files",
                self.opt_dict[self.opt_idx]
            )

            # Load filters
            filters_np = np.load(filter_path)
            self.filters = torch.tensor(filters_np, device='cuda', dtype=torch.float32).unsqueeze(0)

            # Apply propagation and binaural processing
            self.prop = stereo_anechoic(
                self.stim,
                self.lstn_pos_conf[self.pos_conf_idx],
                self.spk_pos,
                self.sofa_path,
                self.fs
            )
            self.bin_stim = self.prop(self.filters)

        except Exception as e:
            print(f"Error processing filters for pos {self.pos_idx}, angle {self.angle_idx}, opt {self.opt_idx}: {e}")
            # Restore old filters to keep the program running
            self.filters = old_filters
    
args = argparse.ArgumentParser(description="Binaural Listening")
args.add_argument('--folder_path', type=str, required=True, help='Path to folder containing config.json and artifacts')
args = args.parse_args()

if __name__ == "__main__":
    
    binaural_listening_instance = binaural_listening(args.folder_path)


    pygame.init()
    window_size = (400, 400)
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                
                # Stimulus selection
                if event.key == pygame.K_r:
                    binaural_listening_instance.select_stim(0)
                    binaural_listening_instance.load_filters()
                if event.key == pygame.K_s:
                    binaural_listening_instance.select_stim(1)
                    binaural_listening_instance.load_filters()
                if event.key == pygame.K_m:
                    binaural_listening_instance.select_stim(2)
                if event.key == pygame.K_p:
                    binaural_listening_instance.select_stim(3)
                    binaural_listening_instance.load_filters()

                # Optimisation target selection
                if pygame.K_0 <= event.key <= pygame.K_9:
                    number_pressed = event.key - pygame.K_0
                    binaural_listening_instance.select_optimisation(number_pressed)
                    binaural_listening_instance.load_filters()

                # Angle selection
                if event.key == pygame.K_PERIOD:
                    binaural_listening_instance.angle_idx += 1
                    binaural_listening_instance.angle_idx %= len(binaural_listening_instance.angle_dict)
                    print(f"Selected angle: {binaural_listening_instance.angle_dict[binaural_listening_instance.angle_idx]}")
                    binaural_listening_instance.load_filters()
                if event.key == pygame.K_COMMA:
                    binaural_listening_instance.angle_idx -= 1
                    binaural_listening_instance.angle_idx %= len(binaural_listening_instance.angle_dict)
                    print(f"Selected angle: {binaural_listening_instance.angle_dict[binaural_listening_instance.angle_idx]}")
                    binaural_listening_instance.load_filters()

                # Listening position selection
                if event.key == pygame.K_LEFT:
                    binaural_listening_instance.pos_idx -= 1
                    binaural_listening_instance.pos_idx %= len(binaural_listening_instance.lstn_pos_conf[binaural_listening_instance.pos_conf_idx])
                    print(f"Selected position: {binaural_listening_instance.pos_idx}")
                elif event.key == pygame.K_RIGHT:
                    binaural_listening_instance.pos_idx += 1
                    binaural_listening_instance.pos_idx %= len(binaural_listening_instance.lstn_pos_conf[binaural_listening_instance.pos_conf_idx])
                    print(f"Selected position: {binaural_listening_instance.pos_idx}")
                
                # Optimised position selection
                elif event.key == pygame.K_UP:  # example for stimulus increment
                    binaural_listening_instance.stim_idx += 1
                    binaural_listening_instance.stim_idx %= 4
                    binaural_listening_instance.select_lstn_conf(binaural_listening_instance.pos_conf_idx)
                    binaural_listening_instance.load_filters()
                elif event.key == pygame.K_DOWN:  # example for stimulus decrement
                    binaural_listening_instance.stim_idx -= 1
                    binaural_listening_instance.stim_idx %= 4
                    binaural_listening_instance.select_lstn_conf(binaural_listening_instance.pos_conf_idx)
                    binaural_listening_instance.load_filters()
                
                elif event.key == pygame.K_SPACE:
                    binaural_listening_instance.listen()
