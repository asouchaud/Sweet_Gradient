from __future__ import annotations

import torch

from scripts.optimisation.nn_init import load_NN


def test_init_scheme_1_places_impulses_on_both_channels() -> None:
    # MultiSourceFIRModule has one branch per loudspeaker. For init=1 the code sets the first
    # sample of each channel to 1 and leaves all other samples at 0, giving an impulse on both.
    model = load_NN(n_samples=4, device="cpu", dtype=torch.float32, init=1)

    left = model[0].weight.detach().flatten()
    right = model[1].weight.detach().flatten()
    expected_per_channel = torch.zeros(4, dtype=torch.float32)
    expected_per_channel[0] = 1.0

    assert torch.equal(left, expected_per_channel)
    assert torch.equal(right, expected_per_channel)


def test_init_scheme_2_keeps_right_channel_silent_initially() -> None:
    # For init=2 the code sets only the first left-channel tap to 1 and keeps the right-channel
    # filter at 0.
    model = load_NN(n_samples=4, device="cpu", dtype=torch.float32, init=2)

    left = model[0].weight.detach().flatten()
    right = model[1].weight.detach().flatten()
    expected_left = torch.zeros(4, dtype=torch.float32)
    expected_left[0] = 1.0

    assert torch.equal(left, expected_left)
    assert torch.equal(right, torch.zeros(4, dtype=torch.float32))


def test_init_scheme_3_starts_from_all_zero_filters() -> None:
    # For init=3 the implementation explicitly fills the whole weight tensor with zeros, so the
    # expected number of non-zero entries is exactly 0.
    model = load_NN(n_samples=4, device="cpu", dtype=torch.float32, init=3)

    for branch in (model[0], model[1]):
        assert torch.count_nonzero(branch.weight.detach()).item() == 0