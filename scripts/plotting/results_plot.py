from math import nan
import torch
import numpy as np
import matplotlib.pyplot as plt

from scripts.plotting.graph_plot import plot_time
from scripts.plotting.graph_plot import plot_freq
from scripts.plotting.graph_plot import plot_stem_signal
from scripts.plotting.graph_plot import plot_frequency_response
from scripts.plotting.dense_arrows_plot import plot_arrows
from scripts.plotting.graph_plot import plot_color


def plot_results(estimation_dict, est_filters, bin_stim, prop_model, loss_dict, target_pos, stim, fs, device, dtype, show=True):
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
    stim_np = torch.as_tensor(stim, device=device, dtype=dtype).detach().cpu().numpy().reshape(-1)
    stimulus_plot, (stimulus_time_ax, stimulus_freq_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 3),
        gridspec_kw={"width_ratios": [2.0, 1.3]},
    )
    plot_stem_signal(
        stim_np,
        ax=stimulus_time_ax,
        title="Input stimulus (time domain)",
        max_min=float(np.max(np.abs(stim_np))) if stim_np.size and np.max(np.abs(stim_np)) > 0.0 else 1.0,
    )
    plot_frequency_response(
        h=stim_np,
        fs=fs,
        ax_mag=stimulus_freq_ax,
        mag_title="Input stimulus (frequency domain)",
        include_phase=False,
    )
    stimulus_plot.tight_layout()

    target_marker = torch.as_tensor(target_pos, device=device, dtype=dtype)

    arrow_plot = None
    loc_model = perceptual_models.get("localisation")
    if "localisation" in estimation_dict and loc_model is not None:
        arrow_plot = plot_arrows(
            prop_model.lstn_pos,
            prop_model.spk_pos,
            torch.tensor(estimation_dict["localisation"], device=device, dtype=dtype),
            torch.tensor(estimation_dict["uncertainty"], device=device, dtype=dtype) if "uncertainty" in estimation_dict else None,
            torch.tensor(estimation_dict["colouration"], device=device, dtype=dtype) if "colouration" in estimation_dict else None,
            target_pos=target_marker,
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
        plt.show()

    return {
        "time_plot": time_plot,
        "freq_plot": freq_plot,
        "stimulus_plot": stimulus_plot,
        "arrow_plot": arrow_plot,
        "color_l_plot": color_l_plot,
        "color_r_plot": color_r_plot,
    }


def plot_stereo_results(estimation_dict, est_filters, bin_stim, stereo_prop, loss_dict, target_pos, stim, fs, device, dtype, show=True):
    """Backward-compatible alias for older call sites."""
    return plot_results(
        estimation_dict=estimation_dict,
        est_filters=est_filters,
        bin_stim=bin_stim,
        prop_model=stereo_prop,
        loss_dict=loss_dict,
        target_pos=target_pos,
        stim=stim,
        fs=fs,
        device=device,
        dtype=dtype,
        show=show,
    )


def plot_pair_directivity_polar(
    est_filters,
    spk_pos,
    fs,
    group_indices,
    radius=None,
    n_angles=720,
    freqs_eval_hz=(250, 500, 1000, 2000, 4000),
    c=343.0,
    overlay_curve=None,
    overlay_curves=None,
    overlay_labels=None,
    show=False,
    db_floor=None,
    overlay_avg_directivity=True,
):
    """
    Plot frequency-dependent directivity for a selected loudspeaker group.

    For each azimuth on a circle, this computes the geometric propagation delay from
    each loudspeaker to the observation point, sums the delayed FIRs, and plots
    10·log10(|H|²) in dB (same scale for every frequency; no per-curve peak normalisation).

    If ``overlay_avg_directivity`` is True (default), a dashed curve overlays the
    **log of the linear average** of those per-band powers:
    ``10·log10(mean_i |H(f_i)|²)``, i.e. same bands as the solid curves, averaged in
    linear power then converted to dB (not the mean of the dB traces).

    ``group_indices`` can contain any number of loudspeaker indices
    (2, 3, 4, ...).

    If ``overlay_curve`` is provided, it must be shape (N, 2) with
    ``[angle_deg, level_db]`` samples; these are overlaid for reference.
    If ``overlay_curves`` is provided, it must contain one or more arrays with
    that same shape.
    If ``overlay_labels`` is provided, it must contain one label per
    overlay curve.
    If ``db_floor`` is ``None``, the lower dB limit is computed from the data
    (including the optional overlay), preserving the existing behavior.
    """
    filters = np.atleast_2d(np.asarray(est_filters, dtype=float))
    spk_pos = np.asarray(spk_pos, dtype=float)

    group_indices = np.asarray(group_indices, dtype=int).reshape(-1)
    if group_indices.size < 1:
        raise ValueError("group_indices must contain at least one loudspeaker index.")

    # Use the centroid of the selected loudspeaker group as local reference origin.
    group_mid = np.mean(spk_pos[group_indices], axis=0)

    if radius is None:
        # Keep a practical default observation radius on the same order as the
        # original setup (around the array listening radius), now centered on
        # the selected loudspeaker-group reference frame.
        radius = float(np.mean(np.linalg.norm(spk_pos[group_indices], axis=1)))

    n_taps = filters.shape[1]
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    freqs_eval_hz = [float(f) for f in freqs_eval_hz]
    mag2_by_freq = np.zeros((len(freqs_eval_hz), len(theta)), dtype=float)

    h_group = filters[group_indices]
    p_group = spk_pos[group_indices]

    # Frequency-domain summation with exact (continuous) phase delays.
    # This avoids jagged artifacts from time-domain fractional-delay realizations.
    n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    omega = 2.0 * np.pi * freqs
    H_group = np.fft.rfft(h_group, n=n_fft, axis=1)
    freq_bin_idx = [int(np.argmin(np.abs(freqs - f_hz))) for f_hz in freqs_eval_hz]
    eps = 1e-12

    for k, ang in enumerate(theta):
        obs = group_mid + np.array([radius * np.cos(ang), radius * np.sin(ang)], dtype=float)
        tau_group = np.linalg.norm(obs - p_group, axis=1) / c
        phase_group = np.exp(-1j * np.outer(tau_group, omega))
        H_sum = np.sum(H_group * phase_group, axis=0)
        for fi, bi in enumerate(freq_bin_idx):
            mag2_by_freq[fi, k] = np.abs(H_sum[bi]) ** 2

    log_mean_mag2 = 10.0 * np.log10(np.mean(mag2_by_freq, axis=0) + eps)

    fig = plt.figure(figsize=(8.5, 6))
    ax = fig.add_subplot(111, projection="polar")
    min_db = float("inf")
    max_db = float("-inf")
    for fi, f_hz in enumerate(freqs_eval_hz):
        dir_db = 10.0 * np.log10(mag2_by_freq[fi] + eps)
        min_db = min(min_db, float(np.min(dir_db)))
        max_db = max(max_db, float(np.max(dir_db)))
        ax.plot(theta, dir_db, linewidth=1.6, label=f"{int(round(f_hz))} Hz")

    if overlay_avg_directivity:
        min_db = min(min_db, float(np.min(log_mean_mag2)))
        max_db = max(max_db, float(np.max(log_mean_mag2)))
        ax.plot(
            theta,
            log_mean_mag2,
            linewidth=2.0,
            linestyle="--",
            color="tab:red",
            label=r"Average over eval. freqs",
            zorder=6,
        )

    overlay_curve_list = []
    if overlay_curve is not None:
        overlay_curve_list.append(overlay_curve)
    if overlay_curves is not None:
        overlay_curve_list.extend(overlay_curves)
    if overlay_labels is not None and len(overlay_labels) != len(overlay_curve_list):
        raise ValueError("overlay_labels must contain one label per overlay curve.")

    overlay_colors = ["k", "tab:purple", "tab:brown", "tab:gray", "tab:olive"]
    for curve_idx, curve in enumerate(overlay_curve_list):
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError("Each overlay curve must have shape (N, 2) as [angle_deg, level_db].")
        overlay_theta = np.deg2rad(np.mod(curve[:, 0], 360.0))
        overlay_db = curve[:, 1]
        min_db = min(min_db, float(np.min(overlay_db)))
        max_db = max(max_db, float(np.max(overlay_db)))
        ax.scatter(
            overlay_theta,
            overlay_db,
            color=overlay_colors[curve_idx % len(overlay_colors)],
            s=20,
            marker="o",
            label=overlay_labels[curve_idx] if overlay_labels is not None else ("Data overlay" if curve_idx == 0 else f"Data overlay {curve_idx + 1}"),
            zorder=5,
        )

    span = max(max_db - min_db, 1e-9)
    pad = 0.05 * span
    y_floor = max(-40.0, min_db - pad) if db_floor is None else float(db_floor)
    y_ceiling = max_db + pad
    ax.set_ylim(y_floor, y_ceiling)
    idx_str = ", ".join(str(int(i)) for i in group_indices.tolist())
    ax.set_title(f"Group directivity (sources {idx_str}), r={radius:.1f} m")
    # Keep legend fully inside figure bounds so saved PNGs don't clip it.
    fig.subplots_adjust(right=0.72)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    ax.grid(True)

    if show:
        plt.show()

    return fig
