from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("sofar")
pytest.importorskip("torchaudio")

from scripts.optimisation.nn_init import load_NN, load_optimiser
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.propagation.stereo_anechoic import stereo_anechoic
from scripts.setup.setup import load_gtfb, load_hrirs, load_stim, set_seed
from scripts.train.train import Train


def test_reference_10_epoch_near_field_run_matches_known_output() -> None:
    # This is a regression test for the exact optimisation configuration provided by the user.
    # It intentionally exercises the real project pipeline end-to-end:
    # setup -> propagation -> perceptual models -> training -> evaluation.
    #
    # The expected values below come directly from the reference output supplied in the prompt for:
    # - near-field mode
    # - 10 epochs
    # - 10 FIR samples
    # - target angle 15 degrees
    # - random stimulus with seed 46
    # - one listener position at [0.0, 0.5]
    #
    # To keep the run deterministic across environments, the test forces CPU execution instead of
    # relying on CUDA/MPS availability.
    repo_root = Path(__file__).resolve().parents[1]
    sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
    idx_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"

    dtype = torch.float32
    device = torch.device("cpu")
    fs = 48000

    near_field = 1
    epochs = 10
    n_samples = 10
    target_angle = 15.0
    stimulus = "random"
    lstn_pos = [[0.0, 0], [0.0, 0.5], [0.0, -0.5]]
    spk_pos = [[1.732, 1.0], [1.732, -1.0]]
    lr = 0.001
    init_type = 1
    reg_type = 2
    gradnorm = 0
    seed = 46

    # The seed is part of the reference configuration and is required because the stimulus and some
    # internal initialisation steps depend on pseudorandom numbers.
    set_seed(seed=seed)

    gtfb = load_gtfb(fs=fs, device=device, dtype=dtype)
    stim = load_stim(fs=fs, device=device, dtype=dtype, selected_stim=stimulus)
    hrirs = load_hrirs(dtype=dtype, device=device, sofa_path=str(sofa_path), idx_path=str(idx_path))

    stereo_prop = stereo_anechoic(
        stim=stim,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        fs=fs,
        sofa_file_path=str(sofa_path),
    )

    loc_model = llado2025(
        loss=F.mse_loss,
        target=target_angle,
        hrirs=hrirs,
        fs=fs,
        gtfb=gtfb,
        lstn_pos_conf=lstn_pos,
    ).to(device=device, dtype=dtype)

    col_model = spectral_matching(
        loss=F.l1_loss,
        target=0,
        stim=stim,
        hrirs=hrirs,
        target_angle=target_angle,
        fs=fs,
        lstn_pos_conf=lstn_pos,
        sofa_path=str(sofa_path),
    ).to(device=device, dtype=dtype)

    # The reference script instantiates desena2020 as well, even though it is not added to the
    # perceptual_dict. We do the same to keep the executed code path as close as possible.
    _unc_model = desena2020(
        loss=F.l1_loss,
        target=0,
        hrirs=hrirs,
        fs=fs,
        gtfb=gtfb,
    ).to(device=device, dtype=dtype)

    perceptual_dict = {
        "localisation": loc_model,
        "colouration": col_model,
    }

    nn_model = load_NN(n_samples=n_samples, device=device, dtype=dtype, init=init_type)
    nn_optimiser = load_optimiser(nn_model, lr=lr)

    trainer = Train(
        n_epoch=epochs,
        nn=nn_model,
        prop=stereo_prop,
        perceptual_dict=perceptual_dict,
        optimiser=nn_optimiser,
        reg=reg_type,
        loggers=[None, None],
        scheduler=None,
        gradnorm=gradnorm,
    )

    trainer.train()
    est_filters, estimation_dict, _bin_stim = trainer.evaluate()

    # These expected FIR coefficients are copied from the user's provided reference output.
    expected_filters = np.array(
        [
            [
                9.9004465e-01,
                -8.9720655e-03,
                1.4326015e-03, 
                9.4809802e-04,
                6.2107487e-04,
                -4.4117944e-04,
                -4.4392698e-04,
                6.5900135e-04,
                5.2877783e-04,
                5.2544044e-04,
            ],
            [
                9.8999721e-01,  
                9.6550267e-03,  
                5.2750986e-03, 
                -1.1142826e-03,
                2.3138886e-03,
                -8.6717901e-04,
                4.6243961e-04,
                -5.0887483e-04,
                6.6803687e-04,  
                4.9884320e-04,
            ],
        ],
        dtype=np.float32,
    )

    # These scalar reference values are also taken directly from the user's provided output.
    expected_localisation = [ 0.625, 16.041666, -18.333334 ]
    expected_colouration = 282.60693359375

    # We use tight numerical tolerances because this is intended as a regression test for a known,
    # deterministic CPU run rather than a broad behavioural test.
    filters_np = est_filters.cpu().numpy() if isinstance(est_filters, torch.Tensor) else est_filters
    np.testing.assert_allclose(filters_np, expected_filters, rtol=1e-6, atol=1e-6)
    loc = estimation_dict["localisation"]
    loc_np = np.asarray(loc)
    np.testing.assert_allclose(loc_np, expected_localisation, rtol=1e-6, atol=1e-6)
    assert float(estimation_dict["colouration"]) == pytest.approx(expected_colouration, abs=1e-6)