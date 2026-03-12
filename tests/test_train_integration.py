from __future__ import annotations

import pytest

pytest.importorskip("torch.utils.tensorboard")

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.optimisation.nn_init import load_NN, load_optimiser
from scripts.train.train import Train


class IdentityPropagation(nn.Module):
    def forward(self, spk_filters: torch.Tensor) -> torch.Tensor:
        # This stand-in propagation module returns the filters unchanged so the integration test can
        # exercise the Train class without needing SOFA files, stimuli, or external acoustics data.
        return spk_filters


class DifferenceModel(nn.Module):
    def __init__(self, target: float) -> None:
        super().__init__()
        self.target = target
        self.loss = F.mse_loss
        self.loss_norm = 1.0
        self.lstn_pos_conf = None

    def forward(self, bin_stim: torch.Tensor) -> torch.Tensor:
        # The "perceptual" estimate here is simply mean(left) - mean(right), chosen because it is
        # differentiable and easy to interpret by hand.
        left = bin_stim[:, 0:1, :].mean(dim=-1)
        right = bin_stim[:, 1:2, :].mean(dim=-1)
        return left - right


def test_train_moves_estimate_towards_target() -> None:
    # This is a lightweight integration test for the actual Train loop:
    # - the project NN is used unchanged,
    # - the optimiser is the real project optimiser,
    # - the Train class performs real forward/backward updates.
    # We start from init=3, which creates all-zero filters, so the initial estimate is exactly 0.
    nn_model = load_NN(n_samples=8, device="cpu", dtype=torch.float32, init=3)
    optimiser = load_optimiser(nn_model, lr=0.05)

    trainer = Train(
        n_epoch=80,
        nn=nn_model,
        prop=IdentityPropagation(),
        perceptual_dict={"localisation": DifferenceModel(target=1.0)},
        optimiser=optimiser,
        reg=0,
        loggers=[None, None],
        scheduler=None,
        gradnorm=0,
    )

    _, initial_estimates, _ = trainer.evaluate()
    trainer.train()
    _, final_estimates, _ = trainer.evaluate()

    initial_error = abs(float(initial_estimates["localisation"]) - 1.0)
    final_error = abs(float(final_estimates["localisation"]) - 1.0)

    # Because the initial filters are all zero and the model computes mean(left) - mean(right),
    # the initial estimate is 0, so the initial error relative to target 1.0 is exactly 1.0.
    # We do not hard-code that number here; instead we check the more important behavioural claim:
    # training should reduce the error.
    assert initial_error > final_error

    # With this simple objective and 80 Adam steps at lr=0.05, the final estimate should end up
    # close to the target. The tolerance of 0.15 is chosen as a robust behavioural bound rather
    # than an exact value, since optimisation trajectories can vary slightly across environments.
    assert 0.15 > final_error