import torch
import torch.nn as nn
import sofar as sf

from scripts.helpers.miscelaneous import set_target

from scripts.helpers.miscelaneous import pos2angle, sofa_find, get_IRs
from scripts.helpers.dsp import causal_conv1d, compute_reference_bin

class interaural_phase_difference_matching(nn.Module):
    """
    Returns an IPD mismatch score per propagated binaural signal.

    Expected input:
        bin_stim: (batch, n_listeners, 2, T)

    Output:
        (batch * n_listeners,)  IPD error (mean absolute wrapped phase error).

    Notes:
        - This module computes a *difference to reference*, so the natural target is 0.
        - Use with ListenerBasedLoss by setting target=0 and loss=F.l1_loss or F.mse_loss.
    """

    def __init__(self, loss, target, target_pos, fs, sofa_path, lstn_pos, n_fft=4096, fmin=200.0, fmax=7000.0, eps=1e-12, device = 'cpu', dtype = torch.float64):
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

        self.target = set_target(self.lstn_pos, target) # target is a scalar for IPD matching, so we pass None for lstn_pos
        self._set_references()
        self.loss = loss
        pi_ = torch.as_tensor(torch.pi, device=self.device, dtype=self.dtype)
        self.loss_norm = self.loss(pi_, torch.zeros_like(pi_)) # The maximum possible loss for normalization is pi for L1 loss and pi^2 for MSE loss


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

        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / float(self.fs)).to(device=ipd_err.device)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        if mask.any():
            ipd_err = ipd_err[:, :, mask]

        # Mean absolute wrapped phase error per signal.
        return torch.mean(torch.abs(ipd_err), dim=-1)
    

    def _set_references(self):
        assert type(self.lstn_pos) == torch.Tensor, "lstn_pos must be a Torch tensor here"
        
        target_angles = pos2angle(self.target_pos, self.lstn_pos)

        src_idx = sofa_find(target_angles, self.sofaile)

        # HRIRs: (n_listeners, 2, ir_len) — 2 = left ear, right ear
        self.reference_hrirs = get_IRs(src_idx, self.sofaile, self.device, self.dtype)


    def _wrap_to_pi(self,x):
        return (x + torch.pi) % (2 * torch.pi) - torch.pi

