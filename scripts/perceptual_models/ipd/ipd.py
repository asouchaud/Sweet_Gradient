import torch
import torch.nn as nn


def _wrap_to_pi(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


class interaural_phase_difference(nn.Module):
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

    supports_listener_indices = False

    def __init__(self, loss, target, reference_stim, fs, n_fft=4096, fmin=200.0, fmax=7000.0, eps=1e-12):
        super().__init__()

        if reference_stim is None:
            raise ValueError("reference_stim is required.")

        self.reference_stim = reference_stim
        self.fs = fs
        self.n_fft = int(n_fft)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.eps = float(eps)

        self.target = target
        self.loss = loss
        pi_ = torch.as_tensor(torch.pi, device=self.reference_stim.device, dtype=self.reference_stim.dtype)
        self.loss_norm = self.loss(pi_, torch.zeros_like(pi_)) # The maximum possible loss for normalization is pi for L1 loss and pi^2 for MSE loss

    def forward(self, bin_stim):
        if len(bin_stim.shape) != 4:
            raise ValueError("Expected bin_stim with shape (batch, n_listeners, 2, T).")
        if bin_stim.shape[2] != 2:
            raise ValueError("Expected bin_stim.shape[2] == 2 (left/right).")

        batch_size, n_lstn_pos, _two, t = bin_stim.shape
        bin_reshaped = bin_stim.view(batch_size * n_lstn_pos, 2, t)

        if self.reference_stim.ndim == 2:
            ref = self.reference_stim.unsqueeze(0).expand(batch_size * n_lstn_pos, 2, t)
        elif self.reference_stim.ndim == 3:
            if self.reference_stim.shape[0] == n_lstn_pos:
                ref = self.reference_stim.repeat(batch_size, 1, 1)
            else:
                raise ValueError("reference_stim and bin_stim disagree on the number of listener positions.")
        else:
            raise ValueError("reference_stim must have shape (2, T) or (n_listeners, 2, T).")

        bin_f = torch.fft.rfft(bin_reshaped, n=self.n_fft, dim=-1)
        ref_f = torch.fft.rfft(ref, n=self.n_fft, dim=-1)

        # IPD := phase(L) - phase(R)
        ipd_bin = _wrap_to_pi(torch.angle(bin_f[:, 0, :]) - torch.angle(bin_f[:, 1, :]))
        ipd_ref = _wrap_to_pi(torch.angle(ref_f[:, 0, :]) - torch.angle(ref_f[:, 1, :]))

        ipd_err = _wrap_to_pi(ipd_bin - ipd_ref)

        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / float(self.fs)).to(device=ipd_err.device)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        if mask.any():
            ipd_err = ipd_err[:, mask]

        # Mean absolute wrapped phase error per signal.
        return torch.mean(torch.abs(ipd_err), dim=-1)

