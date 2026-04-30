import json
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.train import ListenerBasedLoss


def _load_run_experiment():
    module_path = REPO_ROOT / "main_parameter_sweep.py"
    spec = importlib.util.spec_from_file_location("main_parameter_sweep", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load sweep entrypoint at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_experiment"):
        raise ImportError(f"'run_experiment' not found in {module_path}")
    return module.run_experiment


@dataclass
class SweepRun:
    phase: str
    params: dict[str, Any]
    metrics: dict[str, float]
    score: float | None = None


def _normalise(values: list[float]) -> list[float]:
    if len(values) == 0:
        return []
    min_v = float(min(values))
    max_v = float(max(values))
    if math.isclose(min_v, max_v):
        return [0.0 for _ in values]
    span = max_v - min_v
    return [(v - min_v) / span for v in values]


def _apply_balanced_score(runs: list[SweepRun], w_quality: float = 0.6, w_runtime: float = 0.25, w_instability: float = 0.15) -> None:
    quality_error = _normalise([r.metrics["quality_metric"] for r in runs])
    runtime = _normalise([r.metrics["runtime_seconds"] for r in runs])
    instability = _normalise([r.metrics["instability_metric"] for r in runs])
    for i, run in enumerate(runs):
        run.score = -(w_quality * quality_error[i]) - (w_runtime * runtime[i]) - (w_instability * instability[i])


def _weights_candidates() -> list[dict[str, float]]:
    return [
        {"localisation": 1.0, "uncertainty": 0.0, "colouration": 0.0},
        {"localisation": 0.7, "uncertainty": 0.15, "colouration": 0.15},
        {"localisation": 0.5, "uncertainty": 0.25, "colouration": 0.25},
        {"localisation": 0.34, "uncertainty": 0.33, "colouration": 0.33},
        {"localisation": 0.2, "uncertainty": 0.4, "colouration": 0.4},
        {"localisation": 0.15, "uncertainty": 0.7, "colouration": 0.15},
        {"localisation": 0.15, "uncertainty": 0.15, "colouration": 0.7},
    ]


def _loss_builder(loc_model, col_model, unc_model, regularisation, device, dtype):
    return {
        "localisation": ListenerBasedLoss(model=loc_model, weight=1.0, device=device, dtype=dtype),
        "uncertainty": ListenerBasedLoss(model=unc_model, weight=1.0, device=device, dtype=dtype),
        "colouration": ListenerBasedLoss(model=col_model, weight=1.0, device=device, dtype=dtype),
    }


def _run_single(phase: str, env_overrides: dict[str, Any], weight_overrides: dict[str, float] | None) -> SweepRun:
    run_experiment = _load_run_experiment()
    result = run_experiment(
        loss_dict_builder=_loss_builder,
        run_label=phase,
        env_overrides=env_overrides,
        loss_weight_overrides=weight_overrides,
    )
    return SweepRun(
        phase=phase,
        params={"env_overrides": env_overrides, "weight_overrides": weight_overrides or {}},
        metrics={
            "quality_metric": float(result["quality_metric"]),
            "runtime_seconds": float(result["runtime_seconds"]),
            "instability_metric": float(result["instability_metric"]),
        },
    )


def run_parameter_sweep(max_runs: int = 140, output_path: str = "results/parameter_sweep_summary.json") -> list[SweepRun]:
    runs: list[SweepRun] = []

    lr_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    for lr in lr_values:
        if len(runs) >= max_runs:
            break
        runs.append(_run_single("phase1_lr", {"lr": lr}, None))
    _apply_balanced_score(runs)
    top_lr = sorted([r for r in runs if r.phase == "phase1_lr"], key=lambda r: r.score or -1e9, reverse=True)[:2]
    kept_lr = [r.params["env_overrides"]["lr"] for r in top_lr] or [1e-3]

    batch_pairs = [(2, 2), (4, 3), (8, 4), (16, 6)]
    for stim_batch, lstn_batch in batch_pairs:
        for lr in kept_lr:
            if len(runs) >= max_runs:
                break
            runs.append(_run_single("phase2_batch", {"stimulus_batch_size": stim_batch, "listener_batch_size": lstn_batch, "lr": lr}, None))
    _apply_balanced_score(runs)

    structure_grid = [
        {"n_stimulus": n_stim, "num_lstn_pos": n_lstn, "n_filter_taps": taps}
        for n_stim in (8, 16, 24, 32)
        for n_lstn in (5, 9, 13, 17)
        for taps in (16, 32, 64, 128)
    ]
    rng = np.random.default_rng(46)
    sampled_structure = list(rng.choice(structure_grid, size=min(36, len(structure_grid)), replace=False))
    for cfg in sampled_structure:
        if len(runs) >= max_runs:
            break
        runs.append(_run_single("phase3_structure", dict(cfg), None))
    _apply_balanced_score(runs)

    weight_values = _weights_candidates()
    for weights in weight_values:
        if len(runs) >= max_runs:
            break
        runs.append(_run_single("phase4_weights", {}, weights))
    _apply_balanced_score(runs)

    top_overall = sorted(runs, key=lambda r: r.score or -1e9, reverse=True)[:5]
    for best in top_overall:
        for seed in (46, 47, 48):
            if len(runs) >= max_runs:
                break
            env_overrides = dict(best.params["env_overrides"])
            env_overrides["seed"] = seed
            weight_overrides = dict(best.params["weight_overrides"])
            runs.append(_run_single("phase5_confirm", env_overrides, weight_overrides))
    _apply_balanced_score(runs)

    summary = {
        "n_runs": len(runs),
        "top_10": [
            {
                "phase": r.phase,
                "score": r.score,
                "params": r.params,
                "metrics": r.metrics,
            }
            for r in sorted(runs, key=lambda x: x.score or -1e9, reverse=True)[:10]
        ],
        "all_runs": [
            {
                "phase": r.phase,
                "score": r.score,
                "params": r.params,
                "metrics": r.metrics,
            }
            for r in runs
        ],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return runs


if __name__ == "__main__":
    run_parameter_sweep()
