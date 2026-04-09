import torch

from scripts.perceptual_models.desena2020.DEPRACTED_desena2020_helpers import rms


def desena2020_loudnessweighting(interauralcues, distance):
    
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

