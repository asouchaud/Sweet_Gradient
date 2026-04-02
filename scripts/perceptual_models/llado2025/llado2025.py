import torch
import torch.nn as nn

from scripts.perceptual_models.llado2025.llado2025_estimateinterauralcues import llado2025_estimateinterauralcues
from scripts.perceptual_models.llado2025.llado2025_distance import llado2025_distance
from scripts.perceptual_models.llado2025.llado2025_estimatelocalisation import llado2025_estimatelocalisation_torch
from scripts.perceptual_models.llado2025.llado2025_helpers import near_field_target
from scripts.perceptual_models.llado2025.llado2025_helpers import near_field_comp_angles

class llado2025(nn.Module):
    def __init__(self, loss, target_pos, hrirs, fs, gtfb, lstn_pos_conf = None, tau_IACC=0.01, tau_L=1):
        super(llado2025, self).__init__()
        
        self.dtype = hrirs.dtype
        self.device = hrirs.device

        self.fs = fs
        self.gtfb = gtfb
        self.tau_IACC = tau_IACC
        self.tau_L = tau_L
        self.lstn_pos_conf = lstn_pos_conf

        self.template = llado2025_estimateinterauralcues(hrirs.unsqueeze(0), fs, gtfb, tau_IACC)
        self.template['azimuths'] = torch.arange(-90,95,5,dtype=self.dtype, device=self.device).view(-1,1)
        self.template['ild'] = self.template['ild'].squeeze()
        self.template['itd'] = self.template['itd'].squeeze()

        self.target_pos = torch.as_tensor(target_pos, device=self.device, dtype=self.dtype).reshape(-1)
        
        if isinstance(lstn_pos_conf, type(None)):
            self.target = torch.rad2deg(torch.atan2(self.target_pos[1], self.target_pos[0]))
        elif isinstance(lstn_pos_conf, list):
            self.target = near_field_comp_angles(self.target_pos, lstn_pos_conf)
        else:
            print("Wrong data type, defaulting to global target position angle")
            self.target = torch.rad2deg(torch.atan2(self.target_pos[1], self.target_pos[0]))
        self.loss = loss
        self.loss_norm = self.loss(torch.tensor([30.0], device=self.device, dtype=self.dtype), torch.tensor([-30.0], device=self.device, dtype=self.dtype)) # Maximum possible loss for normalization THIS IS FOR A STEREO WITH SPANNING ANGLE OF 60 DEGREE

    def forward(self, bin_stim):
        # Expects bin_stim as (batch, n_listeners, 2, T), where 2 is the left and right ear and T is the number of time samples of each ear signal;
        # returns localisation per listener as (batch, n_listeners).
        
        # Estimate Interaural Cues
        interauralcues = llado2025_estimateinterauralcues(bin_stim, self.fs, self.gtfb, self.tau_IACC)

        # Calculate Distance
        distance = llado2025_distance(interauralcues, self.template)

        # Estimate Localisation
        estimated_localisation = llado2025_estimatelocalisation_torch(distance, self.template, self.tau_L)
        return estimated_localisation
