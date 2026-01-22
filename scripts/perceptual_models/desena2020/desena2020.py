import torch
import torch.nn as nn

from scripts.perceptual_models.desena2020.desena2020_estimateinterauralcues import desena2020_estimateinterauralcues
from scripts.perceptual_models.desena2020.desena2020_distance import desena2020_distance
from scripts.perceptual_models.desena2020.desena2020_loudnessweighting import desena2020_loudnessweighting
from scripts.perceptual_models.desena2020.desena2020_estimateuncertainty import desena2020_estimateuncertainty


class desena2020(nn.Module):
    def __init__(self, hrirs, fs, gtfb, tau_IACC=0.01):
        super(desena2020, self).__init__()
        
        self.dtype = hrirs.dtype
        self.device = hrirs.device

        self.fs = fs
        self.gtfb = gtfb
        self.tau_IACC = tau_IACC

        self.template = desena2020_estimateinterauralcues(hrirs, fs, gtfb, tau_IACC)
        self.template['azimuths'] = torch.arange(-90,95,5,dtype=self.dtype, device=self.device, requires_grad=True).view(-1,1)
        self.template['ild'] = self.template['ild'].squeeze()
        self.template['itd'] = self.template['itd'].squeeze()

        self.loss_norm = 1.0 # No normalization needed for uncertainty as it is already between 0 and 1
        
    def forward(self, bin_stim):
        # Estimate Interaural Cues
        interauralcues = desena2020_estimateinterauralcues(bin_stim, self.fs, self.gtfb, self.tau_IACC)

        # Calculate Distance
        distance = desena2020_distance(interauralcues, self.template)

        # Apply Loudness Weighting
        distance = desena2020_loudnessweighting(interauralcues, distance)

        # Estimate Uncertainty
        uncertainty = desena2020_estimateuncertainty(distance)

        return uncertainty