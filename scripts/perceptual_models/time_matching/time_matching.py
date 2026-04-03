import os

import sofar as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TF

from scripts.perceptual_models.time_matching.time_matching_helpers import near_field_comp_angles
from scripts.perceptual_models.time_matching.time_matching_helpers import time_match
from scripts.setup.helpers import true_conv1d


class time_matching(nn.Module):
    supports_listener_indices = False  ################# TRUE IS BROKEN ################

    def __init__(self, loss, target, stim, hrirs, target_pos, fs, lstn_pos_conf = None, n_fft=4096, sofa_path = os.path.join('data', 'impulses', 'SOFA', 'SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa')):
        super(time_matching, self).__init__()
        
        if stim is None or hrirs is None or fs is None:
            raise ValueError("stim, hrirs, and fs are required.")

        self.dtype = hrirs.dtype
        self.device = hrirs.device

        self.fs = fs
        self.n_fft = n_fft
        self.hrirs = hrirs
        self.stim = stim
        self.sofaile = sf.read_sofa(sofa_path)

        self.target_pos = torch.as_tensor(target_pos, device=self.device, dtype=self.dtype).reshape(-1)
        if self.target_pos.numel() != 2:
            raise ValueError("target_pos must contain exactly two coordinates.")

        if isinstance(lstn_pos_conf, type(None)):
            
            self.target_angle = torch.rad2deg(torch.atan2(self.target_pos[1], self.target_pos[0]))
            self.reference_hrir = self.get_IR(self.sofa_find(self.target_angle)).view(2,1,-1).to(dtype=self.dtype, device=self.device)
            self.reference_stim = true_conv1d(self.stim, self.reference_hrir)

            low_stim = TF.lowpass_biquad(self.reference_stim, self.fs, 7000)

        elif isinstance(lstn_pos_conf, list):

            self.target_angle = near_field_comp_angles(self.target_pos, lstn_pos_conf)

            self.reference_hrir = self.get_IR(self.sofa_find(self.target_angle)).view(2*len(lstn_pos_conf),1,-1).to(dtype=self.dtype, device=self.device)
            self.reference_stim = true_conv1d(self.stim, self.reference_hrir).view(len(lstn_pos_conf), 2, -1)

            low_stim = TF.lowpass_biquad(self.reference_stim, self.fs, 7000)
        else:
            print("Wrong data type, defaulting to global target position angle")
            self.target_angle = torch.rad2deg(torch.atan2(self.target_pos[1], self.target_pos[0]))
            self.reference_hrir = self.get_IR(self.sofa_find(self.target_angle)).view(2,1,-1).to(dtype=self.dtype, device=self.device)
            self.reference_stim = true_conv1d(self.stim, self.reference_hrir)
            low_stim = TF.lowpass_biquad(self.reference_stim, self.fs, 7000)


        self.target = target
        self.loss = loss
        self.loss_norm = time_match(self.reference_stim, low_stim).mean().item() # Maximum possible loss for normalization

    def sofa_find(self, spk_angle):
        # Take out distance coordinate from SOFA source position and add elv = 0 to speaker angles
        spk_angle = torch.as_tensor(spk_angle%360, device=self.device, dtype = self.dtype).view(-1,1)
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

    def forward(self, bin_stim, listener_indices=None):
        # Expects bin_stim as (batch, n_listeners, 2, T), where 2 is the left and right ear and T is the number of time samples of each ear signal;
        # returns flattened coloration scores with shape (batch * n_listeners,).

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

        reference_stim_base = self.reference_stim
        if reference_stim_base.ndim == 2:
            reference_stim_base = reference_stim_base.unsqueeze(0)
        elif listener_indices is not None:  ################# BROKEN ###################
            reference_stim_base = reference_stim_base[listener_indices]

        if reference_stim_base.shape[0] == 1:
            reference_stim = reference_stim_base.expand_as(bin_stim_reshaped)
        else:
            if reference_stim_base.shape[0] != N_lstn_pos:
                raise ValueError("reference_stim and bin_stim disagree on the number of listener positions.")
            reference_stim = reference_stim_base.repeat(batch_size, 1, 1)
        
        coloration = time_match(bin_stim_reshaped, reference_stim)

        return coloration.view(-1)
