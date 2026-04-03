import torch
import torch.nn.functional as F
import numpy as np

from scripts.perceptual_models.llado2025.llado2025_helpers import hilbert_transform
from scripts.perceptual_models.llado2025.llado2025_helpers import softmax_st

from scripts.setup.helpers import true_conv1d


def llado2025_estimateinterauralcues(bin_stim,fs,gtfb, tau_IACC=0.01):
    
    # bin_stim.shape = (batch_size, n_listners, left/right ear, signal_len)
    # gtfb.shape = (n_banks, 1, n_filter_taps) (1 for F.conv1d)

    if len(bin_stim.shape) != 4:
        raise ValueError(f"Expected bin_stim.shape == (batch_size, n_listeners, 2, signal_len), got {tuple(bin_stim.shape)}")
    if bin_stim.shape[2] != 2:
        raise ValueError(f"Expected bin_stim.shape[2] == 2 (left/right ear), got {int(bin_stim.shape[2])}")
    
    batch_size = bin_stim.shape[0]
    N_lstn_pos = bin_stim.shape[1]
    
    N_signals = batch_size*N_lstn_pos
    N_banks = gtfb.shape[0]
    L_signal = bin_stim.shape[-1]
    device = bin_stim.device
    dtype = bin_stim.dtype

    bin_stim_reshaped = bin_stim.view(N_signals, 2, L_signal)

    ## Peripheral Processing
    # Compute Lag values
    maxLag = np.floor(0.001*fs)
    lagValues = torch.arange(-maxLag, maxLag + 1, 1, dtype=dtype, device=device) / fs

    # Gamma Tone Filter Bank Filtering (gtfbout_left/right shape: batch_size, N_banks, signal_len)
    gtfbout_left = true_conv1d(bin_stim_reshaped[:, 0:1, :], gtfb)
    gtfbout_right = true_conv1d(bin_stim_reshaped[:, 1:2, :], gtfb)

    # Half-Wave Rectification Low Frequency (rectified_left/right_LF shape: batch_size, N_banks < 1.5 kHz, signal_len)
    rectified_left_LF = torch.max(gtfbout_left[:,:11,:], torch.zeros_like(gtfbout_left[:,:11,:]))
    rectified_right_LF = torch.max(gtfbout_right[:,:11,:], torch.zeros_like(gtfbout_right[:,:11,:]))

    # Half-Wave Rectification High Frequency (Hilbert Transform) (rectified_left/right_HF shape: batch_size, N_banks > 1.5 kHz, signal_len)
    rectified_left_HF = torch.abs(hilbert_transform(gtfbout_left[:,11:,:]))
    rectified_right_HF = torch.abs(hilbert_transform(gtfbout_right[:,11:,:]))

    # Concatenate Frequency Channels (rectified_left/right shape: batch_size, N_banks, signal_len)
    rectified_left = torch.cat([rectified_left_LF, rectified_left_HF], dim = 1) # rectified_left/right.shape = (N_signals, N_banks, rect_sig_len)
    rectified_right = torch.cat([rectified_right_LF, rectified_right_HF], dim = 1)

    # Format for Torch Conv1d (left_sig shape: 1, N_signals*N_banks, signal_len ¦right_sig shape: N_signals*N_banks, 1, signal_len)
    sig_left = rectified_left.view(1,N_signals*N_banks,-1) # Reshaping done for F.conv1d 
    sig_right = rectified_right.view(N_signals*N_banks,1,-1)

    # Estimate Interaural Cross-Correlation (iaccFuncts shape before reshaping: 1, N_signals*N_banks, padding)
    iaccFuncts = F.conv1d(sig_left, sig_right, padding=int(maxLag), groups=N_signals * N_banks)
    iaccFuncts = iaccFuncts.view(batch_size, N_lstn_pos, N_banks, -1)

    # ITD/ILD Estimation
    itd = torch.sum(torch.mul(softmax_st(iaccFuncts, dim=-1, tau=tau_IACC), lagValues), dim=-1) # Softmax straight-through estimator
    # itd = torch.sum(torch.mul(torch.softmax(iaccFuncts/tau_IACC, dim=-1), lagValues), dim=-1) # Basic Softmax
    ild = (20 * torch.log10(torch.sqrt(torch.sum(gtfbout_right ** 2, dim=-1) + 1e-12)) - 
           20 * torch.log10(torch.sqrt(torch.sum(gtfbout_left ** 2, dim=-1) + 1e-12))) # Adding small constant to avoid log of zero
    
    # Format for desena2020_distance_torch
    itd = itd.view(batch_size, N_lstn_pos, N_banks)
    ild = ild.view(batch_size, N_lstn_pos, N_banks)

    ## Return
    interauralcues = {
        "itd": itd,
        "ild": ild,
        "gtfbout_left": gtfbout_left,
        "gtfbout_right": gtfbout_right,
        "iacc": iaccFuncts
    }
    return interauralcues