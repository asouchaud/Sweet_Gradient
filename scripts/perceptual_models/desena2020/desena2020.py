import torch
import torch.nn as nn


from scripts.helpers.miscelaneous import get_hrirs_for_template, load_gtfb, set_target

from scripts.helpers.peripheral_processing import estimate_interauralcues

from scripts.helpers.template_comparison import itd_ild_template_distance

from scripts.helpers.dsp import rms


class desena2020(nn.Module):
    def __init__(self, loss, target, fs, sofa_path, lstn_pos, tau_IACC=0.01, device = 'cpu', dtype = torch.float64):
        super(desena2020, self).__init__()

        self.name = "desena2020"
        self.ref_flag = 0
        self.fs = fs
        self.device = device
        self.dtype = dtype 
        
        self.tau_IACC = tau_IACC

        self.lstn_pos = lstn_pos

        self.gtfb = load_gtfb(fs, device, dtype)
        hrirs = get_hrirs_for_template(sofa_path, device, dtype)

        self.template = estimate_interauralcues(hrirs.unsqueeze(0), fs, self.gtfb, tau_IACC)
        self.template['azimuths'] = torch.arange(-90,95,5,dtype=self.dtype, device=self.device).view(-1,1)
        self.template['ild'] = self.template['ild'].squeeze()
        self.template['itd'] = self.template['itd'].squeeze()
        
        self.Hmin = 0
        self.Hmin = self._set_Hmin() # Hard coded APPARENT SADIE value

        self.target = set_target(lstn_pos, target)
        self.loss = loss
        self.loss_norm = 1.0 # No normalization needed for uncertainty as it is already between 0 and 1
        

    def forward(self, bin_stim):
        # Expects bin_stim as (batch, n_listeners, 2, T), where 2 is the left and right ear and T is the number of time samples of each ear signal;
        # returns uncertainty per listener as (batch, n_listeners, 1). 
        interauralcues = estimate_interauralcues(bin_stim, self.fs, self.gtfb, self.tau_IACC)

        # Calculate Distance
        distance = itd_ild_template_distance(interauralcues, self.template)

        # Apply Loudness Weighting
        distance = self._loudness_weighting(interauralcues, distance)

        # Estimate Uncertainty
        uncertainty = self._estimate_uncertainty(distance)

        return uncertainty
    

    def _estimate_uncertainty(self, distance):

        weightedLikelihood = distance['weightedLikelihood']
        n_angles = weightedLikelihood.shape[2]
        natural_angles = self.template['azimuths']

        cos = torch.cos(torch.deg2rad(natural_angles)*2).view(1,1,n_angles)
        sin = torch.sin(torch.deg2rad(natural_angles)*2).view(1,1,n_angles)

        alpha = torch.sum(weightedLikelihood*cos, dim = 2)
        beta = torch.sum(weightedLikelihood*sin, dim = 2)

        H = 1 - torch.sqrt(torch.pow(alpha,2) + torch.pow(beta,2))

        Hnorm = (H - self.Hmin)/(1-self.Hmin)

        return(Hnorm)
    

    def _loudness_weighting(self, interauralcues, distance):
        
        gtfbout_left = interauralcues['gtfbout_left']
        gtfbout_right = interauralcues['gtfbout_right']
        likelihood_f = distance['likelihood_f']

        batch_size = likelihood_f.shape[0]
        N_lstn_pos = likelihood_f.shape[1]
        N_template_pos = likelihood_f.shape[2]
        N_banks = likelihood_f.shape[3]
            
        dtype = gtfbout_left.dtype
        device = gtfbout_right.device

        signal_rms = (rms(gtfbout_left,dim=-1) + rms(gtfbout_right,dim=-1))/2
        signal_dB = 20*torch.log10(signal_rms)

        spl_cfs = torch.tensor([
        80.8040974369988, 69.7152778466767, 63.3217162662439, 58.8466830780741,
        55.7072986018014, 53.3475571597242, 51.6321495968390, 50.3987218539694,
        49.9996318902603, 51.0802858850330, 52.3491290304042, 52.0779324379977,
        49.2861519859267, 46.9451007372847, 46.2399898195513, 46.6506947407588,
        48.2807108683239, 51.3531651995338, 55.7893247423705, 59.9041927864145,
        62.6548923203973, 62.8950513166133, 60.1353379120974, 60.1353379120974], 
        dtype=dtype, device=device).view(1,-1)

        spl_cfs_0 = spl_cfs - spl_cfs[0,9]
        signal_dB_0 = signal_dB - signal_dB[:,9].unsqueeze(1)

        dB_diff = signal_dB_0 - spl_cfs_0

        distance['weighting_function_SPL'] = torch.softmax(dB_diff/10, dim=-1)
        einsum = torch.einsum('cmd,cd->cm', likelihood_f.reshape(batch_size*N_lstn_pos, N_template_pos, N_banks), distance['weighting_function_SPL'])
        distance['weightedLikelihood'] = einsum.reshape(batch_size, N_lstn_pos, N_template_pos)

        return distance


    def _set_Hmin(self):

        template_target = self.template.copy()
        template_template = self.template.copy()

        template_target['ild'] = template_target['ild'].unsqueeze(0)
        template_target['itd'] = template_target['itd'].unsqueeze(0)

        distance = itd_ild_template_distance(template_target, template_template)

        distance = self._loudness_weighting(template_target, distance)

        estimated_uncertainty = self._estimate_uncertainty(distance)

        Hmin = torch.min(estimated_uncertainty)

        return(Hmin)
    

        
    

