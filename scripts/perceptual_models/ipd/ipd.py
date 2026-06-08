import torch
import torch.nn as nn
import sofar as sf

from scripts.helpers.miscelaneous import set_target

from scripts.helpers.miscelaneous import pos2angle, sofa_find, get_IRs
from scripts.helpers.dsp import compute_reference_bin, erb_frequency_bin_weights


class interaural_phase_difference_matching(nn.Module):
    """
    Returns an IPD mismatch score per propagated binaural signal.

    Expected input:
        bin_stim: (batch, n_listeners, 2, T)

    Output:
        (batch * n_listeners,)  IPD error (mean absolute or squared wrapped phase error).

    Notes:
        - This module computes a *difference to reference*, so the natural target is 0.
        - Use with ListenerBasedLoss by setting target=0 and loss=F.l1_loss (MAE) or F.mse_loss (MSE).
        - Set ``use_mse=True`` for MSE or ``use_mse=False`` (default) for MAE.
        - Optional ``erb_weighting`` applies the same per-bin ERB weights as spectral_matching.
    """

    def __init__(
        self,
        loss,
        target,
        target_pos,
        fs,
        sofa_path,
        lstn_pos,
        n_fft=4096,
        fmin=50.0,
        fmax=15000.0,
        eps=1e-12,
        erb_weighting=False,
        use_mse=False,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__()

        self.name = "ipd_matching"
        self.ref_flag = 1
        self.fs = fs
        self.device = device
        self.dtype = dtype

        self.sofaile = sf.read_sofa(sofa_path)

        self.lstn_pos = lstn_pos
        self.target_pos = torch.as_tensor(target_pos, device=self.device, dtype=self.dtype).reshape(-1)

        self.n_fft = int(n_fft)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.eps = float(eps)
        self.erb_weighting = erb_weighting
        self.use_mse = use_mse

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

        self.target = set_target(self.lstn_pos, target) # target is a scalar for IPD matching, so we pass None for lstn_pos
        self._set_references()
        self.loss = loss
        pi_ = torch.as_tensor(torch.pi, device=self.device, dtype=self.dtype)
        zero_ = torch.zeros_like(pi_)
        self.loss_norm = self.loss(pi_, zero_) if not self.use_mse else (pi_ ** 2)
        # Maximum possible loss for normalization: pi (MAE) or pi^2 (MSE).


    def forward(self, bin_stim, reference_stim, listener_indices):
        if len(bin_stim.shape) != 4:
            raise ValueError("Expected bin_stim with shape (batch, n_listeners, 2, T).")
        if bin_stim.shape[2] != 2:
            raise ValueError("Expected bin_stim.shape[2] == 2 (left/right).")

        batch_size, n_lstn_pos, _two, t = bin_stim.shape

        reference_bin = compute_reference_bin(reference_stim=reference_stim, 
                                              reference_hrirs=self.reference_hrirs, 
                                              listener_indices=listener_indices).view(batch_size, n_lstn_pos, 2, t)
        

        assert bin_stim.shape == reference_bin.shape, f"Expected bin_stim and reference_bin to have the same shape after reshaping, but got {bin_stim.shape} and {reference_bin.shape}."

        bin_f = torch.fft.rfft(bin_stim, n=self.n_fft, dim=-1)
        ref_f = torch.fft.rfft(reference_bin, n=self.n_fft, dim=-1)

        # IPD := phase(L) - phase(R)
        ipd_bin = self._wrap_to_pi(torch.angle(bin_f[:, :, 0, :]) - torch.angle(bin_f[:, :, 1, :]))
        ipd_ref = self._wrap_to_pi(torch.angle(ref_f[:, :, 0, :]) - torch.angle(ref_f[:, :, 1, :]))

        ipd_err = self._wrap_to_pi(ipd_bin - ipd_ref)
        ipd_err = ipd_err[..., self.freq_bin_mask]
        bin_err = ipd_err ** 2 if self.use_mse else torch.abs(ipd_err)

        if self.erb_weighting:
            weights = self.freq_bin_weights.view(1, 1, -1)
            m = bin_err.shape[-1]
            return torch.sum(bin_err * weights, dim=-1) / m

        return torch.mean(bin_err, dim=-1)
    

    def _set_references(self):
        assert type(self.lstn_pos) == torch.Tensor, "lstn_pos must be a Torch tensor here"
        
        target_angles = pos2angle(self.target_pos, self.lstn_pos)

        src_idx = sofa_find(target_angles, self.sofaile)

        # HRIRs: (n_listeners, 2, ir_len) — 2 = left ear, right ear
        self.reference_hrirs = get_IRs(src_idx, self.sofaile, self.device, self.dtype)


    def _wrap_to_pi(self,x):
        return (x + torch.pi) % (2 * torch.pi) - torch.pi

