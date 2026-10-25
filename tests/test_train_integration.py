from __future__ import annotations

import pytest

pytest.importorskip("torch.utils.tensorboard")

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.datasets.listener_dataset import get_listener_loader
from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.train.train import ListenerBasedLoss, Train


class FilterPairPropagation(nn.Module):
    def forward(self, stim: torch.Tensor, spk_filters: torch.Tensor, listener_indices) -> torch.Tensor:
        # Train calls prop(stimulus, spk_filters, listener_indices). This stand-in places the two
        # source filters on the two ears for every selected listener, so the loop can run without
        # SOFA files or free-field acoustics.
        n_listeners = len(listener_indices)
        return spk_filters.unsqueeze(1).repeat(1, n_listeners, 1, 1)


class DifferenceModel(nn.Module):
    def __init__(self, target: float) -> None:
        super().__init__()
        self.target = torch.tensor([target], dtype=torch.float32)
        self.loss = F.mse_loss
        self.loss_norm = 1.0
        self.ref_flag = 0

    def forward(self, bin_stim: torch.Tensor) -> torch.Tensor:
        # bin_stim is (batch, n_listeners, 2, time). The estimate is mean(left) - mean(right).
        left = bin_stim[:, :, 0, :].mean(dim=-1)
        right = bin_stim[:, :, 1, :].mean(dim=-1)
        return left - right


def _difference_estimate(nn_model, prop, perceptual_model, stim_loader, lstn_loader) -> float:
    device = nn_model[0].weight.device
    dtype = nn_model[0].weight.dtype
    stimulus = next(iter(stim_loader))
    listener_idx, _listener_pos = next(iter(lstn_loader))
    with torch.no_grad():
        spk_filters = nn_model(torch.ones(1, 1, device=device, dtype=dtype))
        bin_stim = prop(stimulus, spk_filters, listener_idx)
        return float(perceptual_model(bin_stim).reshape(-1)[0])


def test_train_moves_estimate_towards_target() -> None:
    # Lightweight integration test for the Train loop: the project FIR and Adam optimiser run real
    # forward/backward updates over the stimulus and listener loaders. Propagation and the
    # perceptual model are stand-ins with the current call signature.
    nn_model = MultiSourceFIR(
        n_samples=8,
        n_sources=2,
    ).to(device="cpu", dtype=torch.float32)

    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[0, 0] = 1.0

    optimiser = torch.optim.Adam(nn_model.parameters(), lr=0.05)
    stim_loader, _stimulus_dataset = get_stim_loader(8000, 1, 1, stimulus_type="impulse", shuffle=False, device="cpu", dtype=torch.float32)
    lstn_loader, _listener_dataset = get_listener_loader(torch.tensor([[0.0, 0.0]], dtype=torch.float32), 1, False)
    prop = FilterPairPropagation()
    difference_model = DifferenceModel(target=1.0)

    trainer = Train(
        n_epoch=80,
        stim_loader=stim_loader,
        lstn_loader=lstn_loader,
        nn=nn_model,
        prop=prop,
        loss_dict={
            "difference": ListenerBasedLoss(
                model=difference_model,
                weight=1.0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
        },
        optimiser=optimiser,
        logger=None,
    )

    initial_estimate = _difference_estimate(nn_model, prop, difference_model, stim_loader, lstn_loader)
    trainer.train()
    final_estimate = _difference_estimate(nn_model, prop, difference_model, stim_loader, lstn_loader)

    initial_error = abs(initial_estimate - 1.0)
    final_error = abs(final_estimate - 1.0)

    # Both source filters start as the same unit impulse, so mean(left) - mean(right) is 0 and the
    # error against target 1 is 1. Training should reduce that error, and 80 Adam steps at lr=0.05
    # should finish within 0.15 of the target.
    assert initial_error > final_error
    assert 0.15 > final_error
