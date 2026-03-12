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
    # - the default `main.py` configuration (far-field, unless explicitly enabled)
    # - 10 epochs
    # - 10 FIR samples
    # - target angle 15 degrees
    # - random stimulus with seed 46
    # - 3 listener positions (main.py lstn_pos_conf=1)
    #
    # To keep the run deterministic across environments, the test forces CPU execution instead of
    # relying on CUDA/MPS availability.
    repo_root = Path(__file__).resolve().parents[1]
    sofa_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48K_24bit_256tap_FIR_SOFA.sofa"
    idx_path = repo_root / "data" / "impulses" / "SOFA" / "D1_48k_hrirs_idx.csv"

    dtype = torch.float32
    device = torch.device("cpu")
    fs = 48000

    epochs = 10
    n_samples = 10
    target_angle = 15.0
    stimulus = "random"
    lstn_pos = [[0.0, 0], [0.0, 0.5], [0.0, -0.5]]
    spk_pos = [[1.732, 1.0], [1.732, -1.0]]
    lr = 0.001
    init_type = 1
    reg_type = 2
    # `main.py` enables GradNorm by default.
    gradnorm = 1
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
    ).to(device=device, dtype=dtype)

    col_model = spectral_matching(
        loss=F.l1_loss,
        target=0,
        stim=stim,
        hrirs=hrirs,
        target_angle=target_angle,
        fs=fs,
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

    # Mirror `main.py`: train using localisation only (colouration is evaluated afterwards).
    perceptual_dict = {"localisation": loc_model}

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
    # Mirror `main.py`: evaluate with the full perceptual set (including colouration),
    # even though training used localisation only.
    full_perceptual_dict = {
        "localisation": loc_model,
        "uncertainty": _unc_model,
        "colouration": col_model,
    }
    est_filters, estimation_dict, _bin_stim = trainer.evaluate(full_perceptual_dict)

    # These expected FIR coefficients are copied from the user's provided reference output.
    expected_filters = np.array(
        [
            [
            1.0108615159988403,
            0.004902565423846245,
            0.0003747918788623804,
            0.00023912532306797802,
            0.0003370926063975692,
            0.0003758000105891377,
            0.00038051412903778255,
            0.0003864263086117804,
            0.0003916759536502516,
            0.0003917338339050114,
            ],
            [
            0.9890888333320618,
            -0.004881629683822393,
            -0.00037801504320442677,
            -0.00020932632171522886,
            -0.00030668729031406343,
            -0.0003522062035461515,
            -0.0003605461694854492,
            -0.00036708073439002037,
            -0.0003727798227599263,
            -0.00037368852891094923,
            ]
        ],
        dtype=np.float32,
    )

    # These scalar reference values are also taken directly from the user's provided output.
    expected_localisation = [ 1.0416666269302368, 20.0, -19.79166603088379 ]
    expected_colouration = 295.8575439453125

    # We use tight numerical tolerances because this is intended as a regression test for a known,
    # deterministic CPU run rather than a broad behavioural test.
    filters_np = est_filters.cpu().numpy() if isinstance(est_filters, torch.Tensor) else est_filters
    np.testing.assert_allclose(filters_np, expected_filters, rtol=1e-4, atol=1e-4)
    loc = estimation_dict["localisation"]
    loc_np = np.asarray(loc)
    np.testing.assert_allclose(loc_np, expected_localisation, rtol=1e-4, atol=1e-4)
    assert float(estimation_dict["colouration"]) == pytest.approx(expected_colouration, abs=1e-4)