from __future__ import annotations

import pytest

pytest.importorskip("torch.utils.tensorboard")

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.train.train import ListenerBasedLoss, SourceBasedLoss, Train


class DummyPropagation(nn.Module):
    def __init__(self, lstn_pos: list[list[float]]) -> None:
        super().__init__()
        self.lstn_pos = torch.tensor(lstn_pos, dtype=torch.float32)
        self.n_listeners = len(lstn_pos)

    def forward(self, spk_filters: torch.Tensor, listener_indices=None) -> torch.Tensor:
        n_listeners = self.n_listeners if listener_indices is None else len(listener_indices)
        return torch.zeros(
            spk_filters.shape[0],
            n_listeners,
            2,
            spk_filters.shape[-1],
            dtype=spk_filters.dtype,
            device=spk_filters.device,
        )


class FixedEstimationModel(nn.Module):
    supports_listener_indices = True

    def __init__(self) -> None:
        super().__init__()
        self.target = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        self.loss = F.l1_loss
        self.loss_norm = 2.0
        self.seen_listener_indices = None

    def forward(self, bin_stim: torch.Tensor, listener_indices=None) -> torch.Tensor:
        self.seen_listener_indices = listener_indices
        return torch.tensor([[20.0, 30.0]], dtype=bin_stim.dtype, device=bin_stim.device)


class SumRegulariser:
    def __init__(self) -> None:
        self.loss_norm = 2.0
        self.seen_filter = None

    def compute_loss(self, source_filter: torch.Tensor) -> torch.Tensor:
        self.seen_filter = source_filter.clone()
        return source_filter.sum()


def make_trainer(
    lstn_pos: list[list[float]],
    listener_batch_size: int,
    symmetric_listener_sampling: bool,
) -> Train:
    nn_model = MultiSourceFIR(n_samples=4, n_sources=1).to(device="cpu", dtype=torch.float32)
    optimiser = torch.optim.Adam(nn_model.parameters(), lr=0.01)

    return Train(
        n_epoch=1,
        nn=nn_model,
        prop=DummyPropagation(lstn_pos),
        loss_dict={},
        optimiser=optimiser,
        loggers=[None, None],
        scheduler=None,
        listener_batch_size=listener_batch_size,
        symmetric_listener_sampling=symmetric_listener_sampling,
    )


def test_listener_based_loss_subsets_target_and_passes_listener_indices() -> None:
    # ListenerBasedLoss should pass listener_indices through to models that support them and should
    # compare only against the matching target subset.
    model = FixedEstimationModel()
    loss_cfg = ListenerBasedLoss(
        model=model,
        weight=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    loss, estimation = loss_cfg.compute_loss(
        bin_stim=torch.zeros(1, 2, 2, 4, dtype=torch.float32),
        h_full=torch.zeros(1, 1, 4, dtype=torch.float32),
        listener_indices=[1, 2],
    )

    assert model.seen_listener_indices == [1, 2]
    assert torch.equal(estimation, torch.tensor([[20.0, 30.0]], dtype=torch.float32))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_listener_based_loss_applies_listener_weights_after_subsetting() -> None:
    # When listener weights are provided, ListenerBasedLoss should compute the weighted mean over the
    # selected listener subset rather than treating every listener equally.
    model = FixedEstimationModel()
    loss_cfg = ListenerBasedLoss(
        model=model,
        weight=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        listener_weights=torch.tensor([1.0, 0.0, 3.0], dtype=torch.float32),
    )

    loss, estimation = loss_cfg.compute_loss(
        bin_stim=torch.zeros(1, 2, 2, 4, dtype=torch.float32),
        h_full=torch.zeros(1, 1, 4, dtype=torch.float32),
        listener_indices=[0, 2],
    )

    assert model.seen_listener_indices == [0, 2]
    assert torch.equal(estimation, torch.tensor([[20.0, 30.0]], dtype=torch.float32))
    assert loss.item() == pytest.approx(1.25, abs=1e-6)


def test_listener_based_loss_batch_scale_uses_listener_weight_mass() -> None:
    # Weighted listener mini-batches should be scaled by their share of the total listener-weight
    # mass, not merely by the number of listeners they contain.
    model = FixedEstimationModel()
    loss_cfg = ListenerBasedLoss(
        model=model,
        weight=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        listener_weights=torch.tensor([1.0, 0.0, 3.0], dtype=torch.float32),
    )

    assert float(loss_cfg.batch_scale(listener_indices=[0, 1], n_listeners=3)) == pytest.approx(0.25)
    assert float(loss_cfg.batch_scale(listener_indices=[2], n_listeners=3)) == pytest.approx(0.75)


def test_source_based_loss_uses_selected_source_filter_only() -> None:
    # SourceBasedLoss should slice out exactly one source filter from h_full and hand only that
    # slice to the regulariser.
    regulariser = SumRegulariser()
    loss_cfg = SourceBasedLoss(regulariser=regulariser, source_idx=1, weight=1.0)

    h_full = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        dtype=torch.float32,
    )

    loss, estimation = loss_cfg.compute_loss(
        bin_stim=torch.zeros(1, 1, 2, 2, dtype=torch.float32),
        h_full=h_full,
    )

    expected_slice = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)

    assert torch.equal(regulariser.seen_filter, expected_slice)
    assert estimation is None
    assert loss.item() == pytest.approx(3.5)


def test_train_sample_listener_indices_keeps_mirrored_pairs_and_centre_when_symmetric() -> None:
    # For symmetric sampling with an odd batch size, the centre listener should be included and
    # mirrored listeners should be selected only as complete pairs.
    trainer = make_trainer(
        lstn_pos=[[0.0, 0.0], [0.0, 0.5], [0.0, -0.5], [0.0, 1.0], [0.0, -1.0]],
        listener_batch_size=3,
        symmetric_listener_sampling=True,
    )

    indices = trainer._sample_listener_indices()

    assert indices == sorted(indices)
    assert len(indices) == 3
    assert 0 in indices

    for pair in ({1, 2}, {3, 4}):
        assert len(pair.intersection(indices)) in (0, 2)


def test_train_build_listener_batches_cover_all_listeners_without_splitting_pairs() -> None:
    # The symmetric batch builder should cover every listener exactly once while never splitting a
    # mirrored pair across different mini-batches.
    trainer = make_trainer(
        lstn_pos=[[0.0, 0.0], [0.0, 0.5], [0.0, -0.5], [0.0, 1.0], [0.0, -1.0]],
        listener_batch_size=3,
        symmetric_listener_sampling=True,
    )

    batches = trainer._build_listener_batches()
    flattened = sorted(idx for batch in batches for idx in batch)

    assert flattened == [0, 1, 2, 3, 4]

    for batch in batches:
        assert batch == sorted(batch)
        assert 3 >= len(batch)
        for pair in ({1, 2}, {3, 4}):
            assert len(pair.intersection(batch)) in (0, 2)


def test_train_rejects_invalid_symmetric_listener_layout() -> None:
    # Symmetric sampling assumes each non-centre listener belongs to a mirrored pair.
    # Three listeners with the same |y| location violate that contract and should be rejected.
    with pytest.raises(ValueError, match="mirrored pairs"):
        make_trainer(
            lstn_pos=[[0.0, 0.5], [0.0, -0.5], [0.0, 0.5]],
            listener_batch_size=2,
            symmetric_listener_sampling=True,
        )