import torch
import numpy as np
import matplotlib.pyplot as plt

from scripts.plotting.graph_plot import plot_time
from scripts.plotting.graph_plot import plot_freq
from scripts.plotting.dense_arrows_plot import plot_arrows
from scripts.plotting.graph_plot import plot_color


def plot_results(estimation_dict, est_filters, bin_stim, prop_model, loss_dict, target_angle, fs, device, dtype, show=True):
    """
    Plot generic multi-source results (time/frequency filters, localisation arrows,
    and colouration comparison).

    This helper consumes the unified loss_dict and extracts perceptual models from it
    (entries with a non-None `.model`) so callers do not need to keep a separate
    perceptual dictionary in sync.
    """
    perceptual_models = {
        key: loss_cfg.model
        for key, loss_cfg in loss_dict.items()
        if hasattr(loss_cfg, "model") and loss_cfg.model is not None
    }

    time_plot = plot_time(est_filters, show=False)
    freq_plot = plot_freq(est_filters, fs, show=False)

    arrow_plot = None
    loc_model = perceptual_models.get("localisation")
    if "localisation" in estimation_dict and loc_model is not None:
        if loc_model.lstn_pos_conf is not None:
            arrow_plot = plot_arrows(
                prop_model.lstn_pos,
                prop_model.spk_pos,
                torch.tensor(estimation_dict["localisation"], device=device, dtype=dtype),
                torch.tensor(estimation_dict["uncertainty"], device=device, dtype=dtype) if "uncertainty" in estimation_dict else None,
                torch.tensor(estimation_dict["colouration"], device=device, dtype=dtype) if "colouration" in estimation_dict else None,
                loc_target=torch.tensor(target_angle),
                show=False,
            )
        else:
            arrow_plot = plot_arrows(
                prop_model.lstn_pos,
                prop_model.spk_pos,
                torch.tensor(estimation_dict["localisation"], device=device, dtype=dtype),
                torch.tensor(estimation_dict["uncertainty"], device=device, dtype=dtype) if "uncertainty" in estimation_dict else None,
                torch.tensor(estimation_dict["colouration"], device=device, dtype=dtype) if "colouration" in estimation_dict else None,
                show=False,
            )

    color_l_plot = None
    color_r_plot = None
    col_model = perceptual_models.get("colouration")
    if col_model is not None:
        color_l_plot, color_r_plot = plot_color(
            bin_stim,
            col_model.reference_stim,
            prop_model.lstn_pos,
            show=False,
        )

    if show:
        import matplotlib.pyplot as plt

        plt.show()

    return {
        "time_plot": time_plot,
        "freq_plot": freq_plot,
        "arrow_plot": arrow_plot,
        "color_l_plot": color_l_plot,
        "color_r_plot": color_r_plot,
    }


def plot_stereo_results(estimation_dict, est_filters, bin_stim, stereo_prop, loss_dict, target_angle, fs, device, dtype, show=True):
    """Backward-compatible alias for older call sites."""
    return plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=bin_stim,
        prop_model=stereo_prop,
        loss_dict=loss_dict,
        target_angle=target_angle,
        fs=fs,
        device=device,
        dtype=dtype,
        show=show,
    )


def plot_pair_directivity_polar(
    est_filters,
    spk_pos,
    fs,
    pair_indices,
    radius=None,
    n_angles=720,
    freqs_eval_hz=(250, 500, 1000, 2000, 4000),
    c=343.0,
    show=False,
):
    """
    Plot frequency-dependent directivity for a selected loudspeaker pair.

    For each azimuth on a circle, this computes the geometric propagation delay from
    each loudspeaker to the observation point, sums the two delayed FIRs, computes
    and displays normalized level (0 dB peak per frequency curve) on a polar plot.
    """
    filters = np.atleast_2d(np.asarray(est_filters, dtype=float))
    spk_pos = np.asarray(spk_pos, dtype=float)

    i0, i1 = pair_indices

    # Use the midpoint of the selected pair as the local reference origin.
    pair_mid = 0.5 * (spk_pos[i0] + spk_pos[i1])

    if radius is None:
        # Keep a practical default observation radius on the same order as the
        # original setup (around the array listening radius), but now centered
        # on the pair midpoint reference frame.
        radius = float(np.mean(np.linalg.norm(spk_pos[[i0, i1]], axis=1)))

    n_taps = filters.shape[1]
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    freqs_eval_hz = [float(f) for f in freqs_eval_hz]
    mag2_by_freq = np.zeros((len(freqs_eval_hz), len(theta)), dtype=float)

    h0 = filters[i0]
    h1 = filters[i1]
    p0 = spk_pos[i0]
    p1 = spk_pos[i1]

    # Frequency-domain summation with exact (continuous) phase delays.
    # This avoids jagged artifacts from time-domain fractional-delay realizations.
    n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    omega = 2.0 * np.pi * freqs
    H0 = np.fft.rfft(h0, n=n_fft)
    H1 = np.fft.rfft(h1, n=n_fft)
    freq_bin_idx = [int(np.argmin(np.abs(freqs - f_hz))) for f_hz in freqs_eval_hz]

    for k, ang in enumerate(theta):
        obs = pair_mid + np.array([radius * np.cos(ang), radius * np.sin(ang)], dtype=float)
        tau0 = np.linalg.norm(obs - p0) / c
        tau1 = np.linalg.norm(obs - p1) / c

        phase0 = np.exp(-1j * omega * tau0)
        phase1 = np.exp(-1j * omega * tau1)
        H_sum = H0 * phase0 + H1 * phase1
        for fi, bi in enumerate(freq_bin_idx):
            mag2_by_freq[fi, k] = np.abs(H_sum[bi]) ** 2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    eps = 1e-12
    min_db = 0.0
    for fi, f_hz in enumerate(freqs_eval_hz):
        dir_db = 10.0 * np.log10(mag2_by_freq[fi] + eps)
        dir_db = dir_db - np.max(dir_db)
        min_db = min(min_db, float(np.min(dir_db)))
        ax.plot(theta, dir_db, linewidth=1.6, label=f"{int(round(f_hz))} Hz")

    ax.set_ylim(max(-40.0, min_db - 1.0), 1.0)
    ax.set_title(f"Pair directivity (sources {i0}, {i1}), r={radius:.1f} m")
    # Keep legend fully inside figure bounds so saved PNGs don't clip it.
    fig.subplots_adjust(right=0.80)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    ax.grid(True)

    if show:
        plt.show()

    return fig
