## Import Libraries

import torch
import torch.nn as nn

import sofar as sf
import numpy as np

from scripts.helpers.dsp import causal_conv1d, calculate_source_angle_2D
from scripts.helpers.miscelaneous import sofa_find, get_IRs

class SingleSourceBinauralFreeFieldPropagation(nn.Module):
    """One source, multiple listeners. Returns binaural (L/R ear) per listener."""

    def __init__(self, lstn_pos, source_pos, sofa_file_path, fs=48000, delay_samples=None, apply_distance_attenuation=False, device = 'cpu', dtype = torch.float64):
        super().__init__()
        
        self.fs = fs
        self.device = device
        self.dtype = dtype
        self.apply_distance_attenuation = apply_distance_attenuation

        self.lstn_pos = torch.tensor(lstn_pos, device=self.device, dtype=self.dtype)
        self.n_listeners = len(self.lstn_pos)
        self.source_pos = torch.tensor(source_pos, device=self.device, dtype=self.dtype)
        self.sofa_file_path = sofa_file_path
        self.sofaile = sf.read_sofa(sofa_file_path)

        # Broadcast source_pos across listeners to get one [dx, dy] vector per listener.
        source_vec = self.source_pos[None, :] - self.lstn_pos
        # Convert each 2D offset vector into a source-listener distance.
        source_distance = torch.linalg.vector_norm(source_vec.float(), dim=-1)
        self.distance_gain = torch.reciprocal(torch.clamp(source_distance, min=1e-6)).to(
            device=self.device,
            dtype=self.dtype,
        )

        # Angle from listener to source (degrees, 0–360)
        source_angle = calculate_source_angle_2D(source_vec)

        # assert 360 >= source_angle >= 0 #TODO: Make this work

        # SOFA index per listener
        src_idx = sofa_find(source_angle, self.sofaile)

        # HRIRs: (n_listeners, 2, ir_len) — 2 = left ear, right ear
        self.hrirs = get_IRs(src_idx, sofaile=self.sofaile, device=self.device, dtype=self.dtype)
        
        # Optional per-listener delay (samples) to align with another source
        if delay_samples is not None:
            # Pad to match stimulus length (same as stereo_anechoic: concat with zeros)
            max_delay_samples = delay_samples[1]
            self.hrirs = torch.nn.functional.pad(self.hrirs, (0, max_delay_samples))
            
            # Roll the HRIRs by the corresponding sample delay
            for i in range(self.n_listeners):
                d = int(delay_samples[0][i])
                if d != 0:
                    self.hrirs[i, 0, :] = torch.roll(self.hrirs[i, 0, :], d, dims=0)
                    self.hrirs[i, 1, :] = torch.roll(self.hrirs[i, 1, :], d, dims=0)

    def _sofa_find(self, source_angle):
        pos = torch.tensor(
            self.sofaile.SourcePosition[:, :2],
            device=self.device,
            dtype=torch.float32,
        )%360
        source_angle = torch.concat(
            (source_angle, torch.zeros_like(source_angle)), dim=1
        ).to(self.device, dtype=torch.float32)
        diff = source_angle[:, None, :] - pos[None, :, :]
        dists = torch.linalg.vector_norm(diff, dim=2)
        return torch.argmin(dists, dim=1)


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

    def forward(self, stim, source_filter, listener_indices):
        """
        stim: (batch, 1, stim_len)
        source_filter: (batch, 1, filter_len)
        Returns: (batch, n_listeners, 2, T) — 2 = L/R ear.
        """
        n_stim = len(stim)
        n_lstn = len(listener_indices)
        hrirs = self.hrirs[listener_indices].reshape(2*n_lstn, 1, -1)

        src_signal = causal_conv1d(stim, source_filter)
        ear_signals = causal_conv1d(src_signal, hrirs).reshape(n_stim, n_lstn, 2, -1)

        if self.apply_distance_attenuation:
            distance_gain = self.distance_gain[listener_indices].view(1, n_lstn, 1, 1)
            ear_signals = ear_signals * distance_gain
        return ear_signals

## Multiple source anechoic class definition (omitting the "MultipleSource" prefix for brevity)
class BinauralFreeFieldPropagation(nn.Module):
    """Generic multi-source anechoic propagation with summed binaural output."""

    def __init__(self, fs, lstn_pos, spk_pos, sofa_file_path, apply_distance_attenuation=False, device = 'cpu', dtype = torch.float64):
        super().__init__()
        
        self.fs = fs
        self.dtype = dtype
        self.device = device
        self.apply_distance_attenuation = apply_distance_attenuation

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
       
        delay_samples = torch.round((tau_sources - min_tau_per_listener[None, :]) * fs).to(torch.int64).detach().cpu().numpy()
        max_delay_samples = int(np.max(delay_samples))
        
        assert isinstance(max_delay_samples, int) 
        assert isinstance(delay_samples, type(np.array([0])))

        self.sources = nn.ModuleList()
        for source_idx in range(self.n_sources):
            source_delay_samples = delay_samples[source_idx]
            self.sources.append(
                SingleSourceBinauralFreeFieldPropagation(
                    lstn_pos=lstn_pos,
                    source_pos=spk_pos[source_idx],
                    sofa_file_path=sofa_file_path,
                    fs=fs,
                    delay_samples=[source_delay_samples, max_delay_samples],
                    apply_distance_attenuation=apply_distance_attenuation,
                    device=device,
                    dtype=dtype,
                )
            )

    def forward(self, stim, spk_filters, listener_indices=None):
        if listener_indices is None:
            listener_indices = torch.arange(self.n_listeners, device=self.device)

        if spk_filters.shape[1] != self.n_sources:
            raise ValueError(
                f"Expected spk_filters with {self.n_sources} sources on dim=1, "
                f"got {spk_filters.shape[1]}"
            )

        binaural_sum = 0
        for source_idx, source in enumerate(self.sources):
            source_filter = spk_filters[:, source_idx:source_idx + 1, :]
            binaural_sum = binaural_sum + source(stim, source_filter, listener_indices=listener_indices)

        return binaural_sum
