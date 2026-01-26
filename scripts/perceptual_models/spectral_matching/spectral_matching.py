import os

import sofar as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TF



class spectral_matching(nn.Module):
    def __init__(self, stim, hrirs, target_angle, fs, n_fft=4096, sofa_path = os.path.join('data', 'impulses', 'SOFA', 'D1_48K_24bit_256tap_FIR_SOFA.sofa')):
        super(spectral_matching, self).__init__()
        
        self.dtype = hrirs.dtype
        self.device = hrirs.device

        self.fs = fs
        self.n_fft = n_fft
        self.target_angle = target_angle
        self.hrirs = hrirs
        self.stim = stim
        self.sofaile = sf.read_sofa(sofa_path)

        self.reference_hrir = self.get_IR(self.sofa_find(self.target_angle)).view(2,1,-1).to(dtype=self.dtype, device=self.device)
        self.reference_stim = F.conv1d(self.stim, torch.flip(self.reference_hrir, dims=(2,)), padding = 'same').permute(1, 0, 2)
        self.reference_fft = 20*torch.log(torch.abs(torch.fft.rfft(self.reference_stim, n=n_fft, dim = -1))).permute(1,0,2)

        low_stim = TF.lowpass_biquad(self.reference_stim, self.fs, 7000)
        low_fft = 20*torch.log(torch.abs(torch.fft.rfft(low_stim, n=n_fft, dim = -1))).permute(1,0,2)
        self.loss_norm = torch.nn.functional.mse_loss(self.reference_fft, low_fft) # Maximum possible loss for normalization

    def sofa_find(self, spk_angle):
        # Take out distance coordinate from SOFA source position and add elv = 0 to speaker angles
        spk_angle = torch.tensor(spk_angle, device=self.device).view(-1,1)
        pos = torch.tensor(self.sofaile.SourcePosition[:,:2],device=self.device)
        spk_angle = torch.concat((spk_angle, torch.zeros_like(spk_angle)), dim=1).to(self.device)

        # Calculate vector between speaker angles and source position angles
        diff = spk_angle[:,None,:] - pos[None,:,:]

        # Calculate vector norm
        dists = torch.linalg.vector_norm(diff, dim = 2)

        # Find minium amongst the vector norms
        spk_idx = torch.argmin(dists, dim = 1)
        
        return spk_idx

    def get_IR(self, spk_idx):
        # Get hrir
        hrir = torch.tensor(self.sofaile.Data_IR[spk_idx.cpu().numpy()], device=self.device)

        return hrir

    def forward(self, bin_stim):
        
        bin_stim_fft = 20*torch.log(torch.abs(torch.fft.rfft(bin_stim, n=self.n_fft, dim = -1)))
        coloration = F.mse_loss(bin_stim_fft, self.reference_fft.expand_as(bin_stim_fft))

        return coloration.view(-1,1)