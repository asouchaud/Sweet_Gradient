## Import Libraries

import torch
import torch.nn as nn

import sofar as sf
import numpy as np

from scripts.setup.helpers import true_conv1d

class SingleSourceBinauralFreeFieldPropagation(nn.Module):
    """One source, multiple listeners. Returns binaural (L/R ear) per listener."""

    def __init__(self, stim, lstn_pos, source_pos, sofa_file_path, fs=48000, delay_samples=None):
        super().__init__()
        self.fs = fs
        self.dtype = stim.dtype
        self.device = stim.device

        self.stim = stim.detach()
        self.lstn_pos = torch.tensor(lstn_pos, device=self.device, dtype=self.dtype)
        self.n_listeners = len(self.lstn_pos)
        self.source_pos = torch.tensor(source_pos, device=self.device, dtype=self.dtype)
        self.sofaile = sf.read_sofa(sofa_file_path)

        # Vector from each listener to source
        source_vec = self.source_pos[None, :] - self.lstn_pos

        # Angle from listener to source (degrees, 0–360)
        source_angle = torch.remainder(
            self._calculate_source_angle_2D(source_vec),
            torch.tensor(360.0, device=self.device)
        ).unsqueeze(1)

        # SOFA index per listener
        src_idx = self._sofa_find(source_angle)

        # HRIRs: (n_listeners, 2, ir_len) — 2 = left ear, right ear
        hrirs = self._get_IRs(src_idx).to(dtype=self.dtype, device=self.device)
        # Pad to match stimulus length (same as stereo_anechoic: concat with zeros)
        self.hrirs = torch.concat(
            (hrirs, torch.zeros_like(hrirs, dtype=self.dtype)), dim=-1
        )

        # Optional per-listener delay (samples) to align with another source
        if delay_samples is not None:
            for i in range(self.n_listeners):
                d = int(delay_samples[i])
                if d != 0:
                    self.hrirs[i, 0, :] = torch.roll(self.hrirs[i, 0, :], d, dims=0)
                    self.hrirs[i, 1, :] = torch.roll(self.hrirs[i, 1, :], d, dims=0)

    def _sofa_find(self, source_angle):
        pos = torch.tensor(
            self.sofaile.SourcePosition[:, :2],
            device=self.device,
            dtype=torch.float32,
        )
        source_angle = torch.concat(
            (source_angle, torch.zeros_like(source_angle)), dim=1
        ).to(self.device, dtype=torch.float32)
        diff = source_angle[:, None, :] - pos[None, :, :]
        dists = torch.linalg.vector_norm(diff, dim=2)
        return torch.argmin(dists, dim=1)

    def _calculate_source_angle_2D(self, vec1):
        vec1_i = vec1[:, 0] + 1j * vec1[:, 1]
        theta = torch.rad2deg(torch.angle(vec1_i))
        return theta.to(self.device)

    def _get_IRs(self, src_idx):
        left_ear_IR = torch.tensor(
            self.sofaile.Data_IR[src_idx.cpu().numpy(), 0],
            device=self.device,
            dtype=self.dtype,
        )
        right_ear_IR = torch.tensor(
            self.sofaile.Data_IR[src_idx.cpu().numpy(), 1],
            device=self.device,
            dtype=self.dtype,
        )
        return torch.stack((left_ear_IR, right_ear_IR), dim=1)

    def forward(self, source_filter, listener_indices=None):
        """
        source_filter: (batch, 1, filter_len)
        Returns: (batch, n_listeners, 2, T) — 2 = L/R ear.
        """
        hrirs = self.hrirs if listener_indices is None else self.hrirs[listener_indices]

        src_signal = true_conv1d(self.stim, source_filter).permute(1, 0, 2)
        L_ear = true_conv1d(src_signal, hrirs[:, 0:1, :])
        R_ear = true_conv1d(src_signal, hrirs[:, 1:2, :])
        return torch.cat([L_ear.unsqueeze(2), R_ear.unsqueeze(2)], dim=2)

## Multiple source anechoic class definition (omitting the "MultipleSource" prefix for brevity)
class BinauralFreeFieldPropagation(nn.Module):
    """Generic multi-source anechoic propagation with summed binaural output."""

    def __init__(self, stim, lstn_pos, spk_pos, sofa_file_path, fs=48000):
        super().__init__()
        self.fs = fs
        self.dtype = stim.dtype
        self.device = stim.device

        self.stim = stim.detach()
        self.lstn_pos = torch.tensor(lstn_pos, device=self.device, dtype=self.dtype)
        self.n_listeners = len(self.lstn_pos)
        self.spk_pos = torch.tensor(spk_pos, device=self.device, dtype=self.dtype)
        if self.spk_pos.ndim != 2 or self.spk_pos.shape[1] != 2:
            raise ValueError("spk_pos must have shape (n_sources, 2)")

        self.n_sources = self.spk_pos.shape[0]
        if self.n_sources < 1:
            raise ValueError("At least one source position is required")

        # Vectors from each listener to each source: (n_sources, n_listeners, 2)
        source_vec = self.spk_pos[:, None, :] - self.lstn_pos[None, :, :]

        # Time of arrival at each listener: (n_sources, n_listeners)
        tau_sources = torch.linalg.vector_norm(source_vec.float(), dim=-1) / 343
        # Keep previous behaviour: farther sources get additional delay.
        min_tau_per_listener = torch.min(tau_sources, dim=0).values
        delay_samples = torch.round((tau_sources - min_tau_per_listener[None, :]) * fs).to(torch.int64)

        self.sources = nn.ModuleList()
        for source_idx in range(self.n_sources):
            source_delay_samples = [int(delay_samples[source_idx, i]) for i in range(self.n_listeners)]
            self.sources.append(
                SingleSourceBinauralFreeFieldPropagation(
                    stim=stim,
                    lstn_pos=lstn_pos,
                    source_pos=spk_pos[source_idx],
                    sofa_file_path=sofa_file_path,
                    fs=fs,
                    delay_samples=source_delay_samples,
                )
            )

    def forward(self, spk_filters, listener_indices=None):
        if spk_filters.shape[1] != self.n_sources:
            raise ValueError(
                f"Expected spk_filters with {self.n_sources} sources on dim=1, "
                f"got {spk_filters.shape[1]}"
            )

        binaural_sum = 0
        for source_idx, source in enumerate(self.sources):
            source_filter = spk_filters[:, source_idx:source_idx + 1, :]
            binaural_sum = binaural_sum + source(source_filter, listener_indices=listener_indices)

        return binaural_sum
