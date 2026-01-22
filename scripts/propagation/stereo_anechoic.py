## Import Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import sofar as sf
import numpy as np

## Stereo anechoic class definition

class stereo_anechoic(nn.Module):
    def __init__(self, stim, lstn_pos, spk_pos, sofa_file_path, fs = 44100):
        super().__init__()
        self.fs = fs
        self.dtype = stim.dtype
        self.device = stim.device

        self.stim = stim.detach()
        self.lstn_pos = torch.tensor(lstn_pos, device=self.device, dtype=self.dtype)
        self.spk_pos = torch.tensor(spk_pos, device=self.device, dtype=self.dtype)
        self.sofaile = sf.read_sofa(sofa_file_path)
        tau_relative = torch.zeros(len(self.lstn_pos), device=self.device)
        
        # Calculate vector from listener to each speaker
        left_vec = self.spk_pos[0, None, :] - self.lstn_pos
        right_vec = self.spk_pos[1, None, :] - self.lstn_pos
        
        # Calculate angles between listener and speakers
        left_spk_angle = torch.remainder(self.calculate_loudspeaker_angle_2D(left_vec), torch.tensor(360.0, device=self.device)).unsqueeze(1)
        right_spk_angle = torch.remainder(self.calculate_loudspeaker_angle_2D(right_vec), torch.tensor(360.0, device=self.device)).unsqueeze(1)
        
        # Find index corresponding to speaker angle in SOFA file
        left_spk_idx = self.sofa_find(left_spk_angle)
        right_spk_idx = self.sofa_find(right_spk_angle)

        # Get HRIRs of corresponding speakers to listeners
        self.hrirs = self.get_IRs(left_spk_idx,right_spk_idx).to(dtype=self.dtype, device=self.device)

        # Calculate time-delay between left/right speaker to listener
        tau_l_spk = torch.linalg.vector_norm(left_vec.float(), dim=-1) / 343
        tau_r_spk = torch.linalg.vector_norm(right_vec.float(), dim=-1) / 343
        tau_relative = tuple((tau_l_spk - tau_r_spk).tolist())

        # Delay HRIRs by  
        for i in range(len(self.lstn_pos)):
            if tau_relative[i] > 0:
                self.hrirs[i, 0, :, :] = torch.roll(self.hrirs[i, 0, :, :], int(np.round(tau_relative[i] * self.fs)), dims=1)
            else:
                self.hrirs[i, 1, :, :] = torch.roll(self.hrirs[i, 1, :, :], int(np.abs(np.round(tau_relative[i] * self.fs))), dims=1)

    def sofa_find(self, spk_angle):
        # Take out distance coordinate from SOFA source position and add elv = 0 to speaker angles
        pos = torch.tensor(self.sofaile.SourcePosition[:,:2],device=self.device)
        spk_angle = torch.concat((spk_angle, torch.zeros_like(spk_angle)), dim=1).to(self.device)

        # Calculate vector between speaker angles and source position angles
        diff = spk_angle[:,None,:] - pos[None,:,:]

        # Calculate vector norm
        dists = torch.linalg.vector_norm(diff, dim = 2)

        # Find minium amongst the vector norms
        spk_idx = torch.argmin(dists, dim = 1)
        
        return spk_idx
    
    def calculate_loudspeaker_angle_2D(self, vec1):
        
        # Project into complex plane
        vec1_i = vec1[:,0] + 1j * vec1[:,1]

        # Find angle via complex plane angle
        theta = torch.rad2deg(torch.angle(vec1_i))

        return theta.to(self.device)

    def get_IRs(self, left_spk_idx, right_spk_idx):
        
        # Get left speaker to left ear IR
        left_spk_left_ear_IR = torch.tensor(self.sofaile.Data_IR[left_spk_idx.cpu().numpy(), 0], device=self.device)
        # Get left speaker to right ear IR
        left_spk_right_ear_IR = torch.tensor(self.sofaile.Data_IR[left_spk_idx.cpu().numpy(), 1], device=self.device)
        # Get right speaker to left ear IR
        right_spk_left_ear_IR = torch.tensor(self.sofaile.Data_IR[right_spk_idx.cpu().numpy(), 0], device=self.device)
        # Get right speaker to right ear IR 
        right_spk_right_ear_IR = torch.tensor(self.sofaile.Data_IR[right_spk_idx.cpu().numpy(), 1], device=self.device)

        # Concatenate over left/right ear
        left_spk_IR = torch.stack((left_spk_left_ear_IR, left_spk_right_ear_IR), dim=1)
        right_spk_IR = torch.stack((right_spk_left_ear_IR, right_spk_right_ear_IR), dim=1)

        # Concatenate over left/right speaker
        hrirs = torch.stack((left_spk_IR, right_spk_IR), dim=1)

        return hrirs

    def forward(self, spk_filters):

        L_spk_filter = spk_filters[:, 0:1, :]
        R_spk_filter = spk_filters[:, 1:2, :]

        L_spk_signal = F.conv1d(self.stim,L_spk_filter, padding = 'same').permute(1, 0, 2)
        R_spk_signal = F.conv1d(self.stim, R_spk_filter, padding = 'same').permute(1, 0, 2)

        L_ear_signals = F.conv1d(L_spk_signal, torch.flip(self.hrirs[:, 0, 0:1, :],dims=(2,)), padding = 'same') + F.conv1d(R_spk_signal, torch.flip(self.hrirs[:, 1, 0:1, :], dims=(2,)), padding = 'same')
        R_ear_signals = F.conv1d(L_spk_signal, torch.flip(self.hrirs[:, 0, 1:2, :],dims=(2,)), padding = 'same') + F.conv1d(R_spk_signal, torch.flip(self.hrirs[:, 1, 1:2, :],dims=(2,)), padding = 'same')

        bin_stim = torch.cat([L_ear_signals.unsqueeze(2), R_ear_signals.unsqueeze(2)], dim=2)

        return bin_stim

