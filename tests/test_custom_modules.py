from __future__ import annotations

import torch

from scripts.optimisation.custom_modules import MultiSourceFIR, SymmetricFIR


def test_symmetric_fir_weight_property_returns_mirrored_filter() -> None:
    # SymmetricFIR stores only the left half plus the centre tap.
    # The public weight property should reconstruct the full mirrored FIR.
    fir = SymmetricFIR(n_taps=5)

    with torch.no_grad():
        fir.half_weight.copy_(torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32))

    expected = torch.tensor([[1.0], [2.0], [3.0], [2.0], [1.0]], dtype=torch.float32)

    assert torch.equal(fir.weight, expected)


def test_symmetric_fir_set_filter_initialisation_projects_to_symmetric_weights() -> None:
    # A non-symmetric initial filter should be projected onto the symmetric subspace by averaging
    # it with its time-reversed version.
    fir = SymmetricFIR(n_taps=5)

    fir.set_filter_initialisation(
        torch.tensor([[1.0], [2.0], [9.0], [4.0], [5.0]], dtype=torch.float32)
    )

    expected = torch.tensor([[3.0], [3.0], [9.0], [3.0], [3.0]], dtype=torch.float32)

    assert torch.allclose(fir.weight, expected)


def test_multisourcefir_copies_init_weight_to_each_branch_and_stacks_outputs() -> None:
    # MultiSourceFIR should initialise every branch with the same filter and stack the per-branch
    # outputs on a dedicated source dimension.
    init_weight = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    model = MultiSourceFIR(n_samples=3, n_sources=2, init_weight=init_weight)

    x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y = model(x)

    expected_per_source = torch.tensor(
        [[1.0, -2.0, 0.5], [2.0, -4.0, 1.0]],
        dtype=torch.float32,
    )

    assert y.shape == (2, 2, 3)
    assert torch.allclose(y[:, 0, :], expected_per_source)
    assert torch.allclose(y[:, 1, :], expected_per_source)
    assert model[0] is model.branches[0]