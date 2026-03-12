import os

import sofar as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TF

from scripts.perceptual_models.spectral_matching.spectral_matching_helpers import near_field_target
from scripts.perceptual_models.spectral_matching.spectral_matching_helpers import near_field_comp_angles
from scripts.perceptual_models.spectral_matching.spectral_matching_helpers import log_spectral_distortion
from scripts.setup.helpers import true_conv1d


class spectral_matching(nn.Module):
    def __init__(self, loss, target, stim, hrirs, target_angle, fs, lstn_pos_conf = None, n_fft=4096, sofa_path = os.path.join('data', 'impulses', 'SOFA', 'D1_48K_24bit_256tap_FIR_SOFA.sofa')):
        super(spectral_matching, self).__init__()
        
        self.dtype = hrirs.dtype
        self.device = hrirs.device

        self.fs = fs
        self.n_fft = n_fft
        self.target_angle = target_angle
        self.hrirs = hrirs
        self.stim = stim
        self.sofaile = sf.read_sofa(sofa_path)

        if isinstance(lstn_pos_conf, type(None)):
            
            self.reference_hrir = self.get_IR(self.sofa_find(self.target_angle)).view(2,1,-1).to(dtype=self.dtype, device=self.device)
            self.reference_stim = true_conv1d(self.stim, self.reference_hrir)

            low_stim = TF.lowpass_biquad(self.reference_stim, self.fs, 7000)

        elif isinstance(lstn_pos_conf, list):

            near_field_loc = near_field_target(target_angle)
            self.target_angle = near_field_comp_angles(near_field_loc, lstn_pos_conf)

            self.reference_hrir = self.get_IR(self.sofa_find(self.target_angle)).view(2*len(lstn_pos_conf),1,-1).to(dtype=self.dtype, device=self.device)
            self.reference_stim = true_conv1d(self.stim, self.reference_hrir)

            low_stim = TF.lowpass_biquad(self.reference_stim, self.fs, 7000)
        else:
            print("Wrong data type, defaulting for far-field")
            self.target = target_angle


        self.target = target
        self.loss = loss
        self.loss_norm = log_spectral_distortion(self.reference_stim, low_stim).item() # Maximum possible loss for normalization

    def sofa_find(self, spk_angle):
        # Take out distance coordinate from SOFA source position and add elv = 0 to speaker angles
        spk_angle = torch.tensor(spk_angle, device=self.device, dtype = self.dtype).view(-1,1)
        pos = torch.tensor(self.sofaile.SourcePosition[:,:2],device=self.device, dtype = self.dtype)
        spk_angle = torch.concat((spk_angle, torch.zeros_like(spk_angle)), dim=1).to(self.device).to(self.dtype)

        # Calculate vector between speaker angles and source position angles
        diff = spk_angle[:,None,:] - pos[None,:,:]

        # Calculate vector norm
        dists = torch.linalg.vector_norm(diff, dim = 2)

        # Find minium amongst the vector norms
        spk_idx = torch.argmin(dists, dim = 1)
        
        return spk_idx

    def get_IR(self, spk_idx):
        # Get hrir
        hrir = torch.tensor(self.sofaile.Data_IR[spk_idx.cpu().numpy()], device=self.device, dtype = self.dtype)

        return hrir

    def forward(self, bin_stim):

        if len(bin_stim.shape) == 4:
            batch_size = bin_stim.shape[0]
            N_lstn_pos = bin_stim.shape[1]

        elif len(bin_stim.shape) == 3:
            batch_size = bin_stim.shape[0]
            N_lstn_pos = 1
        
        elif len(bin_stim.shape) == 2:
            batch_size = 1
            N_lstn_pos = 1

        else:
            print('Expected input of dimensions greater than 1 and lower than 5')
            return(0)
    
        N_signals = batch_size*N_lstn_pos
        L_signal = bin_stim.shape[-1]
        bin_stim_reshaped = bin_stim.view(N_signals, 2, L_signal)
        
        coloration = log_spectral_distortion(bin_stim_reshaped, self.reference_stim.expand_as(bin_stim_reshaped))

        return coloration.view(-1)
