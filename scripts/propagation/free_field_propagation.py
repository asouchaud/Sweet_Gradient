## Import Libraries

import torch
import torch.nn as nn

from scripts.propagation.anechoic_source import AnechoicSourcePropagation


## Stereo anechoic class definition

class BinauralFreeFieldPropagation(nn.Module):
    """Generic multi-source anechoic propagation with summed binaural output."""

    def __init__(self, stim, lstn_pos, spk_pos, sofa_file_path, fs=48000):
        super().__init__()
        self.fs = fs
        self.dtype = stim.dtype
        self.device = stim.device

        self.stim = stim.detach()
        self.lstn_pos = torch.tensor(lstn_pos, device=self.device, dtype=self.dtype)
        self.spk_pos = torch.tensor(spk_pos, device=self.device, dtype=self.dtype)
        if self.spk_pos.ndim != 2 or self.spk_pos.shape[1] != 2:
            raise ValueError("spk_pos must have shape (n_sources, 2)")

        self.n_sources = self.spk_pos.shape[0]
        if self.n_sources < 1:
            raise ValueError("At least one source position is required")

        n_listeners = len(self.lstn_pos)

        # Vectors from each listener to each source: (n_sources, n_listeners, 2)
        source_vec = self.spk_pos[:, None, :] - self.lstn_pos[None, :, :]

        # Time of arrival at each listener: (n_sources, n_listeners)
        tau_sources = torch.linalg.vector_norm(source_vec.float(), dim=-1) / 343
        # Keep previous behaviour: farther sources get additional delay.
        min_tau_per_listener = torch.min(tau_sources, dim=0).values
        delay_samples = torch.round((tau_sources - min_tau_per_listener[None, :]) * fs).to(torch.int64)

        self.sources = nn.ModuleList()
        for source_idx in range(self.n_sources):
            source_delay_samples = [int(delay_samples[source_idx, i]) for i in range(n_listeners)]
            self.sources.append(
                AnechoicSourcePropagation(
                    stim=stim,
                    lstn_pos=lstn_pos,
                    source_pos=spk_pos[source_idx],
                    sofa_file_path=sofa_file_path,
                    fs=fs,
                    delay_samples=source_delay_samples,
                )
            )

    def forward(self, spk_filters):
        if spk_filters.shape[1] != self.n_sources:
            raise ValueError(
                f"Expected spk_filters with {self.n_sources} sources on dim=1, "
                f"got {spk_filters.shape[1]}"
            )

        binaural_sum = 0
        for source_idx, source in enumerate(self.sources):
            source_filter = spk_filters[:, source_idx:source_idx + 1, :]
            binaural_sum = binaural_sum + source(source_filter)

        return binaural_sum
