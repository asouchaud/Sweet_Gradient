## Import Libraries

import torch
import torch.nn as nn
import numpy as np

from scripts.propagation.anechoic_source import anechoic_source


## Stereo anechoic class definition

class stereo_anechoic(nn.Module):
    """Two sources (L/R speakers): uses two anechoic_source instances and sums binaural outputs."""

    def __init__(self, stim, lstn_pos, spk_pos, sofa_file_path, fs=48000):
        super().__init__()
        self.fs = fs
        self.dtype = stim.dtype
        self.device = stim.device

        self.stim = stim.detach()
        self.lstn_pos = torch.tensor(lstn_pos, device=self.device, dtype=self.dtype)
        self.spk_pos = torch.tensor(spk_pos, device=self.device, dtype=self.dtype)
        n_listeners = len(self.lstn_pos)

        # Vectors from each listener to each speaker
        left_vec = self.spk_pos[0, None, :] - self.lstn_pos
        right_vec = self.spk_pos[1, None, :] - self.lstn_pos

        # Time-delay difference (seconds): positive = left speaker farther
        tau_l_spk = torch.linalg.vector_norm(left_vec.float(), dim=-1) / 343
        tau_r_spk = torch.linalg.vector_norm(right_vec.float(), dim=-1) / 343
        tau_relative = (tau_l_spk - tau_r_spk).cpu().numpy()

        # Per-listener delay in samples: delay the farther source so both align at the listener
        left_delay_samples = [
            int(np.round(max(0, tau_relative[i]) * fs)) for i in range(n_listeners)
        ]
        right_delay_samples = [
            int(np.round(max(0, -tau_relative[i]) * fs)) for i in range(n_listeners)
        ]

        self.left_source = anechoic_source(
            stim=stim,
            lstn_pos=lstn_pos,
            source_pos=spk_pos[0],
            sofa_file_path=sofa_file_path,
            fs=fs,
            delay_samples=left_delay_samples,
        )
        self.right_source = anechoic_source(
            stim=stim,
            lstn_pos=lstn_pos,
            source_pos=spk_pos[1],
            sofa_file_path=sofa_file_path,
            fs=fs,
            delay_samples=right_delay_samples,
        )

    def forward(self, spk_filters):
        L_spk_filter = spk_filters[:, 0:1, :]
        R_spk_filter = spk_filters[:, 1:2, :]

        L_bin = self.left_source(L_spk_filter)
        R_bin = self.right_source(R_spk_filter)

        return L_bin + R_bin
