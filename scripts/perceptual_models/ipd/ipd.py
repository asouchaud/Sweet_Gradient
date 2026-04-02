import torch
import torch.nn as nn


def _wrap_to_pi(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


class interaural_phase_difference(nn.Module):
    """
    Returns an IPD mismatch score per propagated binaural signal.

    Expected input:
        bin_stim: (batch, n_listeners, 2, T) or (batch, 2, T) or (2, T)

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
        self.loss_norm = 1.0

    def _reshape_bin(self, bin_stim):
        if len(bin_stim.shape) == 4:
            batch_size = bin_stim.shape[0]
            n_lstn_pos = bin_stim.shape[1]
            t = bin_stim.shape[-1]
            return bin_stim.view(batch_size * n_lstn_pos, 2, t), batch_size, n_lstn_pos
        if len(bin_stim.shape) == 3:
            batch_size = bin_stim.shape[0]
            t = bin_stim.shape[-1]
            return bin_stim.view(batch_size, 2, t), batch_size, 1
        if len(bin_stim.shape) == 2:
            t = bin_stim.shape[-1]
            return bin_stim.view(1, 2, t), 1, 1
        raise ValueError("Expected bin_stim with 2 to 4 dimensions.")

    def _expand_reference(self, reference_stim_base, batch_size, n_lstn_pos, t):
        # Accept reference_stim shaped (2, T) or (n_listeners, 2, T).
        if reference_stim_base.ndim == 2:
            ref = reference_stim_base.unsqueeze(0)  # (1, 2, T)
            ref = ref.expand(batch_size * n_lstn_pos, 2, t)
            return ref

        if reference_stim_base.ndim == 3:
            if reference_stim_base.shape[0] == 1:
                ref = reference_stim_base.expand(batch_size * n_lstn_pos, 2, t)
                return ref
            if reference_stim_base.shape[0] != n_lstn_pos:
                raise ValueError("reference_stim and bin_stim disagree on the number of listener positions.")
            # Repeat listeners for each batch element.
            ref = reference_stim_base.repeat(batch_size, 1, 1)  # (batch*n_listeners, 2, T)
            return ref

        raise ValueError("reference_stim must have shape (2, T) or (n_listeners, 2, T).")

    def forward(self, bin_stim):
        bin_reshaped, batch_size, n_lstn_pos = self._reshape_bin(bin_stim)
        t = bin_reshaped.shape[-1]

        ref = self._expand_reference(self.reference_stim, batch_size, n_lstn_pos, t)

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

