from __future__ import annotations

import torch

from scripts.optimisation.nn_init import load_NN


def test_init_scheme_1_places_impulses_on_both_channels() -> None:
    # In this project, the linear layer stores both loudspeaker FIRs in a single vector of length
    # 2 * n_samples. For init=1 the code sets the first sample of each channel to 1 and leaves all
    # other samples at 0, giving an impulse on both loudspeakers.
    model = load_NN(n_samples=4, device="cpu", dtype=torch.float32, init=1)

    weights = model[0].weight.detach().flatten()
    expected = torch.zeros(8, dtype=torch.float32)
    expected[0] = 1.0
    expected[4] = 1.0

    assert torch.equal(weights, expected)


def test_init_scheme_2_keeps_right_channel_silent_initially() -> None:
    # For init=2 the code sets only the first left-channel tap to 1 and keeps the right-channel
    # filter at 0. This test mirrors that exact layout for n_samples=4.
    model = load_NN(n_samples=4, device="cpu", dtype=torch.float32, init=2)

    weights = model[0].weight.detach().flatten()
    expected = torch.zeros(8, dtype=torch.float32)
    expected[0] = 1.0

    assert torch.equal(weights, expected)


def test_init_scheme_3_starts_from_all_zero_filters() -> None:
    # For init=3 the implementation explicitly fills the whole weight tensor with zeros, so the
    # expected number of non-zero entries is exactly 0.
    model = load_NN(n_samples=4, device="cpu", dtype=torch.float32, init=3)

    weights = model[0].weight.detach()

    assert torch.count_nonzero(weights).item() == 0