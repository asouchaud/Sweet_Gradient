import os

import sofar as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TF

from scripts.helpers.miscelaneous import pos2angle, sofa_find, get_IRs, set_target
from scripts.helpers.dsp import (
    causal_conv1d,
    compute_reference_bin,
    erb_frequency_bin_weights,
)


class spectral_matching(nn.Module):
    supports_listener_indices = True

    def __init__(self, loss, target, target_pos, fs, sofa_path, lstn_pos, n_fft=4096, fmin=50.0, fmax=15000.0, eps = 1e-8, erb_weighting=True, normalize_log_spectrum=False, device = 'cpu', dtype = torch.float64):
        super(spectral_matching, self).__init__()


        self.name = "spectral_matching"
        self.ref_flag = 1
        self.fs = fs
        self.device = device
        self.dtype = dtype 

        self.eps = eps
        self.n_fft = int(n_fft)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.erb_weighting = erb_weighting
        self.normalize_log_spectrum = normalize_log_spectrum
        self.sofaile = sf.read_sofa(sofa_path)

        self.lstn_pos = lstn_pos
        self.target_pos = torch.as_tensor(target_pos, device=self.device, dtype=self.dtype).reshape(-1)
        if self.target_pos.numel() != 2:
            raise ValueError("target_pos must contain exactly two coordinates.")

        fft_frequencies = torch.fft.rfftfreq(self.n_fft, d=1.0 / float(fs)).to(device=device, dtype=dtype)
        freq_bin_mask = (fft_frequencies >= self.fmin) & (fft_frequencies <= self.fmax)
        if not freq_bin_mask.any():
            raise ValueError(
                f"No FFT bins in [{self.fmin}, {self.fmax}] Hz for fs={fs} and n_fft={self.n_fft}."
            )
        self.register_buffer("freq_bin_mask", freq_bin_mask)

        if self.erb_weighting:
            freq_bin_weights = erb_frequency_bin_weights(fft_frequencies[freq_bin_mask])
            self.register_buffer("freq_bin_weights", freq_bin_weights)
        else:
            self.register_buffer("freq_bin_weights", None)

        reference_stim, low_stim = self._set_references()

        self.target = set_target(self.lstn_pos, target)
        self.loss = loss
        self.loss_norm = self._log_spectral_distortion(reference_stim, low_stim).mean().item() # Maximum possible loss for normalization
    

    def forward(self, bin_stim, reference_stim, listener_indices):
        # Expects bin_stim as (batch, n_listeners, 2, T), where 2 is the left and right ear and T is the number of time samples of each ear signal;
        # returns coloration scores with shape (batch, n_listeners) for ListenerBasedLoss (target subset + expand_as).

        if len(bin_stim.shape) != 4:
            raise ValueError(
                f"bin_stim must have shape (batch, n_listeners, 2, T); got ndim={len(bin_stim.shape)}, shape={tuple(bin_stim.shape)}."
            )
        n_stim = bin_stim.shape[0]
        n_lstn = bin_stim.shape[1]
        L_signal = bin_stim.shape[-1]
        
        bin_stim_reshaped = bin_stim.view(n_stim*n_lstn, 2, L_signal)

        reference_bin = compute_reference_bin(reference_stim=reference_stim, 
                                              reference_hrirs=self.reference_hrirs, 
                                              listener_indices=listener_indices)

        colouration = self._log_spectral_distortion(bin_stim_reshaped, reference_bin) #TODO: Normalise both between 0 and 1 to not have difference in level

        return colouration.view(n_stim, n_lstn)


    def _set_references(self):
        assert type(self.lstn_pos) == torch.Tensor, "lstn_pos must be a Torch tensor here"
        
        target_angles = pos2angle(self.target_pos, self.lstn_pos)

        src_idx = sofa_find(target_angles, self.sofaile)

        # HRIRs: (n_listeners, 2, ir_len) — 2 = left ear, right ear
        self.reference_hrirs = get_IRs(src_idx, self.sofaile, self.device, self.dtype)

        reference_stim = causal_conv1d(torch.randn(1,1,self.fs, device=self.device, dtype = self.dtype), self.reference_hrirs.view(len(self.lstn_pos)*2,1,-1)).view(len(self.lstn_pos),2, -1)
        low_stim = TF.lowpass_biquad(reference_stim, self.fs, 7000)

        return(reference_stim, low_stim)


    def _log_spectral_distortion(self, x, y):
        """
        Compute LSD between two time signals.

        Args:
            x, y: tensors of shape [a, b, c]
            n_fft: FFT size (default: c)

        Returns:
            LSD tensor of shape [a]
        """

        assert len(x.shape) == 3
        assert len(y.shape) == 3

        # FFT along time dimension
        X = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        Y = torch.fft.rfft(y, n=self.n_fft, dim=-1)

        # Magnitude
        X_mag = torch.abs(X)
        Y_mag = torch.abs(Y)

        # Log magnitude (dB)
        log_X = 20 * torch.log10(X_mag + self.eps)
        log_Y = 20 * torch.log10(Y_mag + self.eps)

        # Normalise FFTs by peak log magnitude per ear so LSD reflects spectral shape.
        if self.normalize_log_spectrum:
            norm_X = torch.max(torch.max(torch.abs(log_X), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
            norm_Y = torch.max(torch.max(torch.abs(log_Y), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
            log_X = log_X / norm_X.clamp(min=self.eps)
            log_Y = log_Y / norm_Y.clamp(min=self.eps)

        sq_err = (log_X - log_Y) ** 2
        sq_err = sq_err[..., self.freq_bin_mask]

        if self.erb_weighting:
            weights = self.freq_bin_weights.view(1, 1, -1)
            m = sq_err.shape[-1]
            mean_sq = torch.sum(sq_err * weights, dim=-1) / m
        else:
            mean_sq = torch.mean(sq_err, dim=-1)

        lsd_freq = torch.sqrt(mean_sq)  # [a, b]

        # Average across b
        lsd = lsd_freq.mean(dim=-1)  # [a]

        return lsd
