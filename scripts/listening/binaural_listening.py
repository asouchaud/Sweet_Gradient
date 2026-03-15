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

from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation


class binaural_listening:
    def __init__(self, folder_path):
        json_file_path = os.path.join(folder_path, "config.json")
        json_data = json.load(open(json_file_path))
        
        # Loudspkeaker and listener positions
        self.fs = json_data["fs"]
        self.spk_pos = [[1,1], [1,-1]]
        lstn_pos_conf = [[[0.0, 0.0]],  # 1 listner position
                    [[0.0, 0], [0.0, 0.5], [0.0, -0.5]], # 3 listener positions
                    [[0.0, 0], [0.0, 0.5], [0.0, -0.5], [0.5, 0], [-0.5, 0]], # 5 listener positions
                    [[0.0, 0], [0.0, 0.5], [0.0, -0.5], [0.5, 0], [-0.5, 0], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]] # 9
        
        match = re.search(r"n(\d+)", os.path.basename(folder_path))
        number = match.group(1)

        lstn_pos_conf_dict = {1: lstn_pos_conf[0], 3: lstn_pos_conf[1], 5: lstn_pos_conf[2], 9: lstn_pos_conf[3]}
        self.lstn_pos = lstn_pos_conf_dict[int(number)]

        # self.sofa_path = json_data["HRTF"]
        self.sofa_path = r"data\impulses\SOFA\D1_48K_24bit_256tap_FIR_SOFA.sofa"
        
        self.stim = load_stim(self.fs, device='cuda', dtype=torch.float32, selected_stim=json_data["stimulus"])

        self.prop = BinauralFreeFieldPropagation(self.stim, self.lstn_pos, self.spk_pos, self.sofa_path, self.fs)

        filter_path = os.path.join(folder_path, "artifacts", "est_filters.npy")
        filters_np = np.load(filter_path)
        self.filters = torch.tensor(filters_np, device='cuda', dtype=torch.float32).unsqueeze(0)

        self.bin_stim = self.prop(self.filters)
        self.pos_idx = 0
        self.stim_idx = 0
        self.is_playing = 0

    def listen(self):


        if self.is_playing:
            sd.stop()
            self.is_playing = False
            return

        # Otherwise → start playback
        bin_stim_np = self.bin_stim.cpu().detach().numpy().squeeze()
        audio = bin_stim_np[self.pos_idx].T
        audio_norm = audio / np.max(np.abs(audio))

        self.current_audio = audio_norm
        sd.play(self.current_audio, self.fs)
        self.is_playing = True
    
    def select_stim(self, stim_idx):

        stim_dict = {0: "random", 1: "sine", 2: "male_speech", 3: "pop_music"}
        self.stim = load_stim(self.fs, device='cuda', dtype=torch.float32, selected_stim=stim_dict[stim_idx])
        self.prop = BinauralFreeFieldPropagation(self.stim, self.lstn_pos, self.spk_pos, self.sofa_path, self.fs)
        self.bin_stim = self.prop(self.filters)
        print(f"Selected stimulus: {stim_dict[stim_idx]}")

    
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
                if event.key == pygame.K_LEFT:
                    binaural_listening_instance.pos_idx -= 1
                    binaural_listening_instance.pos_idx %= len(binaural_listening_instance.lstn_pos)
                    print(f"Selected position: {binaural_listening_instance.lstn_pos[binaural_listening_instance.pos_idx]}")
                elif event.key == pygame.K_RIGHT:
                    binaural_listening_instance.pos_idx += 1
                    binaural_listening_instance.pos_idx %= len(binaural_listening_instance.lstn_pos)
                    print(f"Selected position: {binaural_listening_instance.lstn_pos[binaural_listening_instance.pos_idx]}")
                
                elif event.key == pygame.K_UP:  # example for stimulus increment
                    binaural_listening_instance.stim_idx += 1
                    binaural_listening_instance.stim_idx %= 4
                    binaural_listening_instance.select_stim(binaural_listening_instance.stim_idx)
                elif event.key == pygame.K_DOWN:  # example for stimulus decrement
                    binaural_listening_instance.stim_idx -= 1
                    binaural_listening_instance.stim_idx %= 4
                    binaural_listening_instance.select_stim(binaural_listening_instance.stim_idx)
                
                elif event.key == pygame.K_SPACE:
                    binaural_listening_instance.listen()

