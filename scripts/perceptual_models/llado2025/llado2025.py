import torch
import torch.nn as nn

from scripts.setup.helpers import load_gtfb

from scripts.perceptual_models.llado2025.llado2025_estimateinterauralcues import llado2025_estimateinterauralcues
from scripts.perceptual_models.llado2025.llado2025_distance import llado2025_distance
from scripts.perceptual_models.llado2025.llado2025_estimatelocalisation import llado2025_estimatelocalisation_torch
from scripts.perceptual_models.llado2025.llado2025_helpers import near_field_comp_angles
from scripts.perceptual_models.llado2025.llado2025_helpers import llado_get_hrirs
from scripts.perceptual_models.llado2025.llado2025_helpers import llado_set_target
from scripts.perceptual_models.llado2025.llado2025_helpers import llado_set_norm


class llado2025(nn.Module):
    def __init__(self, loss, target_pos, fs, sofa_path, lstn_pos = None, spk_pos = None, tau_IACC=0.01, tau_L=1, device = 'cpu', dtype = torch.float64):
        super(llado2025, self).__init__()
        
        self.fs = fs
        self.device = device
        self.dtype = dtype
        
        self.tau_IACC = tau_IACC
        self.tau_L = tau_L

        self.lstn_pos = lstn_pos
        self.spk_pos = spk_pos

        hrirs = llado_get_hrirs(sofa_path, device, dtype)
        self.gtfb = load_gtfb(fs, device, dtype)

        self.template = llado2025_estimateinterauralcues(hrirs.unsqueeze(0), fs, self.gtfb, tau_IACC)
        self.template['azimuths'] = torch.arange(-90,95,5,dtype=self.dtype, device=self.device).view(-1,1)
        self.template['ild'] = self.template['ild'].squeeze()
        self.template['itd'] = self.template['itd'].squeeze()

        self.target_pos, self.target = llado_set_target(lstn_pos, target_pos, device, dtype)
        self.loss = loss
        self.loss_norm = llado_set_norm(self.loss, self.spk_pos, device=self.device, dtype=self.dtype)

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
