import torch
from torch import nn


class SymmetricFIR(nn.Module):
    """An FIR module whose filter coefficients are constrained to be symmetric."""

    def __init__(self, n_taps, bias=False):
        super().__init__()
        if bias:
            raise ValueError("SymmetricFIRModule does not support bias.")
        if n_taps % 2 == 0:
            raise ValueError("SymmetricFIRModule expects an odd number of taps.")

        self.n_taps = n_taps
        self.half_len = n_taps // 2 + 1
        self.half_weight = nn.Parameter(torch.zeros(self.half_len, 1))

    @property
    def weight(self):
        left = self.half_weight[:-1]
        centre = self.half_weight[-1:]
        return torch.cat([left, centre, torch.flip(left, dims=[0])], dim=0)

    def set_filter_initialisation(self, init_weight: torch.Tensor) -> None:
        """Project a full filter to the symmetric parameterisation."""
        symmetric_weight = 0.5 * (init_weight + torch.flip(init_weight, dims=[0]))
        with torch.no_grad():
            self.half_weight.copy_(symmetric_weight[:self.half_len])

    def forward(self, x):
        return x @ self.weight.T


class MultiSourceFIR(nn.Module):
    """One FIR (Linear) layer per source; forward returns (batch, source, taps)."""

    def __init__(self, n_samples, n_sources, init_weight=None):
        super().__init__()
        self.n_sources = n_sources
        self.n_samples = n_samples
        self.branches = nn.ModuleList([
            nn.Linear(1, n_samples, bias=False)
            for _ in range(n_sources)
        ])

        if init_weight is not None:
            for branch in self.branches:
                if hasattr(branch, 'set_filter_initialisation'):
                    branch.set_filter_initialisation(init_weight)
                elif hasattr(branch, 'weight'):
                    self.set_filter_initialisation(branch.weight, init_weight)

    def set_filter_initialisation(self, weight: torch.Tensor, init_weight: torch.Tensor) -> None:
        """Copy the provided initialisation vector into the branch weights in-place."""
        with torch.no_grad():
            weight.copy_(init_weight.reshape_as(weight))

    def forward(self, x):
        # Expects x shaped (batch, 1); returns stacked FIRs shaped (batch, n_sources, n_samples).
        return torch.stack([b(x) for b in self.branches], dim=1)

    def __getitem__(self, i):
        """So downstream can use nn[0], nn[-1] for device/dtype or first/last branch."""
        return self.branches[i]