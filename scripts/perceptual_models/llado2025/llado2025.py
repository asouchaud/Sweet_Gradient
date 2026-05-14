import torch
import torch.nn as nn
import numpy as np 

from scripts.helpers.dsp import softmax_st

from scripts.helpers.miscelaneous import get_hrirs_for_template
from scripts.helpers.miscelaneous import pos2angle
from scripts.helpers.miscelaneous import load_gtfb

from scripts.helpers.peripheral_processing import estimate_interauralcues

from scripts.helpers.template_comparison import itd_ild_template_distance
from scripts.helpers.interaural_cue_grad_balance import InterauralCueGradBalance


class llado2025(nn.Module):
    def __init__(
        self,
        loss,
        target_pos,
        fs,
        sofa_path,
        lstn_pos,
        spk_pos=None,
        tau_IACC=0.01,
        tau_L=1,
        balance_interaural_cue_grads=True,
        device="cpu",
        dtype=torch.float64,
    ):
        super(llado2025, self).__init__()
        
        self.name = "llado2025"
        self.ref_flag = 0
        self.fs = fs
        self.device = device
        self.dtype = dtype
        
        self.tau_IACC = tau_IACC
        self.tau_L = tau_L

        self.lstn_pos = lstn_pos
        self.spk_pos = spk_pos

        hrirs = get_hrirs_for_template(sofa_path, device, dtype)
        self.gtfb = load_gtfb(fs, device, dtype)

        self.template = estimate_interauralcues(hrirs.unsqueeze(0), fs, self.gtfb, tau_IACC)
        self.template['azimuths'] = torch.arange(-90,95,5,dtype=self.dtype, device=self.device).view(-1,1)
        self.template['ild'] = self.template['ild'].squeeze()
        self.template['itd'] = self.template['itd'].squeeze()

        self.target_pos, self.target = self._set_target(target_pos)
        self.loss = loss
        self.loss_norm = self._set_norm()
        self.balance_interaural_cue_grads = balance_interaural_cue_grads
    

    def forward(self, bin_stim):
        # Expects bin_stim as (batch, n_listeners, 2, T), where 2 is the left and right ear and T is the number of time samples of each ear signal;
        # returns localisation per listener as (batch, n_listeners).
        
        # Estimate Interaural Cues
        interauralcues = estimate_interauralcues(bin_stim, self.fs, self.gtfb, self.tau_IACC)
        if self.balance_interaural_cue_grads and torch.is_grad_enabled():
            itd_b, ild_b = InterauralCueGradBalance.apply(interauralcues["itd"], interauralcues["ild"])
            interauralcues = {**interauralcues, "itd": itd_b, "ild": ild_b}
        if torch.is_grad_enabled():
            interauralcues["itd"].retain_grad()
            interauralcues["ild"].retain_grad()
        self.latest_interauralcues = interauralcues

        # Calculate Distance
        distance = itd_ild_template_distance(interauralcues, self.template)

        # Estimate Localisation
        estimated_localisation = self._estimate_localisation(distance)
        
        return estimated_localisation
    
        
    def _estimate_localisation(self, distance):

        # Find most likely direction of arrival per frequency band
        estimated_localisation_f = torch.sum(torch.mul(softmax_st(distance['likelihood_f'],dim=2, tau = self.tau_L), self.template['azimuths']), dim=2) # Softmax straight-through estimator
        # estimated_localisation_f = torch.sum(torch.mul(torch.softmax(distance['likelihood_f']/tau_L,dim=2),template['azimuths']), dim=2) # Basic Softmax
    
        # Average most likely direction of arrival over frequency band
        estimated_localisation = torch.mean(estimated_localisation_f, dim=2)

        return estimated_localisation


    def _set_target(self, target_pos):

        target_pos = torch.as_tensor(target_pos, device=self.device, dtype=self.dtype)
        
        target = pos2angle(target_pos, self.lstn_pos) # TODO: Should the target angles match the available angles in the SOFA files e.g. target at 0.3 deg but 0.0 only available in SOFA
        
        return target_pos, target


    def _set_norm(self):

        if isinstance(self.spk_pos, type(None)):
            return(torch.tensor([1], device=self.device, dtype=self.dtype))
        
        spk_pos = self.spk_pos.detach().cpu().numpy()

        angles = np.rad2deg(np.arctan2(spk_pos[:,1], spk_pos[:,0]))
        angles = np.sort(angles)

        # Compute circular differences (including wrap-around)
        diffs = np.diff(angles)
        wrap_diff = (angles[0] + 360) - angles[-1]
        diffs = np.append(diffs, wrap_diff)

        # Take smallest equivalent angle (important!)
        diffs = np.minimum(diffs, 360 - diffs)

        max_angle_diff = np.max(diffs)
        
        norm = self.loss(torch.tensor([max_angle_diff]), torch.tensor([0.0])) # Maximum possible loss for normalization

        return norm.to(device=self.device, dtype=self.dtype)


