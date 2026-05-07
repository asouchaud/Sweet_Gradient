import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from scripts.datasets.listener_dataset import get_listener_loader
from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.ipd.ipd import interaural_phase_difference_matching
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.setup.setup import build_eval_objects, set_seed
from scripts.train.train import Train


def _compute_perceptual_error_metrics(estimation_dict, eval_loss_dict):
    per_attribute_error = {}
    all_listener_errors = []

    for key, estimation in estimation_dict.items():
        if estimation is None:
            continue
        loss_cfg = eval_loss_dict.get(key)
        if loss_cfg is None or not hasattr(loss_cfg, "model") or not hasattr(loss_cfg.model, "target"):
            continue

        est = np.asarray(estimation, dtype=float).reshape(-1)
        target = loss_cfg.model.target.detach().cpu().numpy().astype(float).reshape(-1)
        n = min(est.size, target.size)
        if n == 0:
            continue

        listener_error = np.abs(est[:n] - target[:n])
        per_attribute_error[key] = float(np.mean(listener_error))
        all_listener_errors.extend(listener_error.tolist())

    if not per_attribute_error:
        return 0.0, 0.0, {}

    quality_error = float(np.mean(list(per_attribute_error.values())))
    instability_error = float(np.std(np.asarray(all_listener_errors, dtype=float)))
    return quality_error, instability_error, per_attribute_error


def run_experiment(
    loss_dict_builder,
    run_label,
    env_overrides=None,
    loss_weight_overrides=None,
):
    repo_root = Path(__file__).resolve().parent
    d = 100
    fs = 48000
    env = SimpleNamespace(
        epochs=200,
        num_lstn_pos=9,
        listener_batch_size=3,
        symmetric_listener_sampling=False,
        selected_stim="white_noise",
        stimulus_batch_size=4,
        n_stimulus=16,
        n_filter_taps=32,
        target_pos=[d, float(np.tan(np.deg2rad(0.0)) * d)],
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        lr=0.01,
        seed=46,
        log=0,
        fs=fs,
    )

    if env_overrides:
        for key, value in env_overrides.items():
            setattr(env, key, value)

    env.n_samples = env.n_filter_taps
    env.dtype = torch.float64
    env.device_str = "cpu"
    env.device = torch.device(env.device_str)

    env.lstn_pos = torch.tensor(
        [[0.0, float(y)] for y in np.linspace(-0.5, 0.5, env.num_lstn_pos)],
        device=env.device,
        dtype=env.dtype,
    )
    base_angle_rad = np.deg2rad(30.0)
    env.speaker_distance = 2.0
    env.spk_pos = torch.tensor(
        [
            [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
            [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
        ],
        device=env.device,
        dtype=env.dtype,
    )
    env.n_sources = len(env.spk_pos)
    set_seed(seed=env.seed)

    stim_loader, _ = get_stim_loader(
        env.fs,
        env.stimulus_batch_size,
        env.n_stimulus,
        stimulus_type=env.selected_stim,
        device=env.device,
        dtype=env.dtype,
    )
    lstn_loader, _ = get_listener_loader(env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling)

    stereo_prop = BinauralFreeFieldPropagation(
        fs=env.fs,
        lstn_pos=env.lstn_pos,
        spk_pos=env.spk_pos,
        sofa_file_path=env.sofa_path,
        device=env.device,
        dtype=env.dtype,
    )
    loc_model = llado2025(loss=F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, device=env.device, dtype=env.dtype)
    col_model = spectral_matching(loss=F.l1_loss, target=0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype)
    unc_model = desena2020(loss=F.l1_loss, target=0, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype)
    _ = interaural_phase_difference_matching(loss=F.l1_loss, target=0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype)
    regularisation = second_moment_regularisation(n_samples=env.n_filter_taps, device=env.device, dtype=env.dtype)
    loss_dict = loss_dict_builder(loc_model, col_model, unc_model, regularisation, env.device, env.dtype)
    if loss_weight_overrides:
        for loss_name, weight in loss_weight_overrides.items():
            if loss_name in loss_dict:
                loss_dict[loss_name].weight = weight

    nn_model = MultiSourceFIR(n_samples=env.n_filter_taps, n_sources=env.n_sources).to(device=env.device, dtype=env.dtype)
    centre_tap_idx = env.n_filter_taps // 2
    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[centre_tap_idx] = 1.0
    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

    trainer = Train(
        n_epoch=env.epochs,
        stim_loader=stim_loader,
        lstn_loader=lstn_loader,
        nn=nn_model,
        prop=stereo_prop,
        loss_dict=loss_dict,
        optimiser=nn_optimiser,
        logger=None,
    )

    start = time.time()
    trainer.train()
    env.time_train_seconds = time.time() - start

    eval_lstn_pos, eval_loss_dict, eval_stim_loader, _ = build_eval_objects(env)
    _, estimation_dict, _ = trainer.evaluate(
        eval_lstn_pos=eval_lstn_pos,
        eval_stimulus_data_loader=eval_stim_loader,
        loss_dict=eval_loss_dict,
    )

    quality_metric, instability_metric, per_attribute_error = _compute_perceptual_error_metrics(
        estimation_dict=estimation_dict,
        eval_loss_dict=eval_loss_dict,
    )
    return {
        "run_label": run_label,
        "quality_metric": quality_metric,
        "instability_metric": instability_metric,
        "runtime_seconds": float(env.time_train_seconds),
        "estimation_dict": estimation_dict,
        "per_attribute_error": per_attribute_error,
    }
