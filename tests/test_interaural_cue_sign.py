from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.helpers.peripheral_processing import estimate_interauralcues
from scripts.setup.setup import load_gtfb


@pytest.fixture
def gtfb_cpu():
    return load_gtfb(48_000, device="cpu", dtype=torch.float64)


def test_ild_positive_when_left_ear_louder(gtfb_cpu) -> None:
    fs = 48_000
    length = 4096
    t = torch.arange(length, dtype=torch.float64) / fs
    tone = 0.5 * torch.sin(2 * torch.pi * 500.0 * t)
    left = 2.0 * tone
    right = 0.25 * tone
    bin_stim = torch.stack([left, right], dim=0).unsqueeze(0).unsqueeze(0)

    cues = estimate_interauralcues(bin_stim, fs, gtfb_cpu, tau_IACC=1e-2)
    ild_mean = cues["ild"].mean().item()

    assert ild_mean > 0.5


def test_itd_sign_left_ear_leads_right_ear(gtfb_cpu) -> None:
    fs = 48_000
    length = 4096
    delay_samples = 20
    impulse = torch.zeros(length, dtype=torch.float64)
    impulse[100] = 1.0
    left = impulse
    right = torch.roll(impulse, delay_samples)
    bin_stim = torch.stack([left, right], dim=0).unsqueeze(0).unsqueeze(0)

    cues = estimate_interauralcues(bin_stim, fs, gtfb_cpu, tau_IACC=1e-2)
    itd_mean = cues["itd"].mean().item()

    assert itd_mean > 0.0
