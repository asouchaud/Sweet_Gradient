from math import nan
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D


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


def plot_arrows_with_directivity(
    est_filters,
    spk_pos,
    lstn_pos,
    loc,
    fs,
    group_idx,
    unc=None,
    col=None,
    target_pos=None,
    vbap_values=None,
    show=True,
    n_angles=360,
    freqs_eval_hz=(250, 500, 1000, 2000, 4000),
    c=343.0,
    directivity_mode="avg",
    directivity_freq_idx=2,
    directivity_scale=0.5,
    directivity_floor_db=-20,
    overlay_curve=None,
    overlay_curves=None,
    overlay_labels=None,
):
    # -----------------------------
    # Prepare inputs
    # -----------------------------
    if unc is None:
        unc = torch.zeros_like(loc)
    if col is None:
        col = torch.zeros_like(loc)

    lstn_pos = lstn_pos.detach().cpu().numpy().squeeze().reshape(-1, 2)
    spk_pos = spk_pos.detach().cpu().numpy().squeeze()
    loc = loc.detach().cpu().numpy().squeeze().reshape(-1)
    unc = unc.detach().cpu().numpy().squeeze().reshape(-1)
    col = col.detach().cpu().numpy().squeeze().reshape(-1)

    # -----------------------------
    # Overlay handling
    # -----------------------------
    overlay_curve_list = []
    if overlay_curve is not None:
        overlay_curve_list.append(overlay_curve)
    if overlay_curves is not None:
        overlay_curve_list.extend(overlay_curves)

    if overlay_labels is not None and len(overlay_labels) != len(overlay_curve_list):
        raise ValueError("overlay_labels must contain one label per overlay curve.")

    # -----------------------------
    # Global mean reference (for alignment)
    # -----------------------------
    all_means = []
    for curve in overlay_curve_list:
        curve = np.asarray(curve, dtype=float)
        all_means.append(np.mean(curve[:, 1]))

    target_mean_db = np.mean(all_means) if len(all_means) > 0 else 0.0

    # -----------------------------
    # Create plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(spk_pos[:, 0], spk_pos[:, 1], c='black', marker='x')
    ax.scatter(lstn_pos[:, 0], lstn_pos[:, 1], c='blue', label='Listeners')

    # -----------------------------
    # Arrows
    # -----------------------------
    arrow_len = 0.2
    cmap = cm.viridis
    norm = colors.Normalize(vmin=0, vmax=18.5)

    for x, y, ang, leng, col_val in zip(lstn_pos[:, 0], lstn_pos[:, 1], loc, unc, col):
        dx = (1 - leng) * arrow_len * np.cos(np.deg2rad(ang))
        dy = (1 - leng) * arrow_len * np.sin(np.deg2rad(ang))
        color = cmap(norm(max(col_val, 1e-3)))

        ax.arrow(x, y, dx, dy,
                 head_width=0.05, head_length=0.08,
                 fc=color, ec=color)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="LSD")

    # -----------------------------
    # Directivity loop
    # -----------------------------
    for i, g in enumerate(group_idx):

        group_indices = np.asarray(g, dtype=int).reshape(-1)

        filters = np.atleast_2d(np.asarray(est_filters, dtype=float))
        h_group = filters[group_indices]
        p_group = spk_pos[group_indices]

        group_mid = np.mean(p_group, axis=0)

        theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

        n_taps = h_group.shape[1]
        n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
        omega = 2*np.pi*freqs
        H_group = np.fft.rfft(h_group, n=n_fft, axis=1)

        freq_bin_idx = [np.argmin(np.abs(freqs - f)) for f in freqs_eval_hz]

        mag2 = np.zeros((len(freqs_eval_hz), len(theta)))

        radius = np.mean(np.linalg.norm(p_group - group_mid, axis=1))

        for k, ang in enumerate(theta):
            obs = group_mid + np.array([radius*np.cos(ang), radius*np.sin(ang)])
            tau = np.linalg.norm(obs - p_group, axis=1) / c
            phase = np.exp(-1j * np.outer(tau, omega))
            H_sum = np.sum(H_group * phase, axis=0)

            for fi, bi in enumerate(freq_bin_idx):
                mag2[fi, k] = np.abs(H_sum[bi])**2

        dir_db = 10*np.log10(mag2 + 1e-12)

        if directivity_mode == "avg":
            dir_use = 10*np.log10(np.mean(mag2, axis=0) + 1e-12)
        else:
            dir_use = dir_db[directivity_freq_idx]

        # -----------------------------
        # Align directivity mean
        # -----------------------------
        dir_mean = np.mean(dir_use)
        dir_aligned = dir_use - dir_mean + target_mean_db
        dir_clipped = np.maximum(dir_aligned, directivity_floor_db)

        r = directivity_scale * (dir_clipped - directivity_floor_db) / (0 - directivity_floor_db)

        # -----------------------------
        # Angle mask
        # -----------------------------
        if len(overlay_curve_list) > 0:
            curve_ref = np.asarray(overlay_curve_list[0])
            start_angle = np.min(np.mod(curve_ref[:, 0], 360.0))
            end_angle = np.max(np.mod(curve_ref[:, 0], 360.0))
        else:
            start_angle, end_angle = 0, 360

        if i == 0:
            off_set = 180 - start_angle
            start_angle += off_set
            end_angle += off_set
        else:
            off_set = 0.0

        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)

        theta_wrapped = np.mod(theta, 2*np.pi)
        start_rad = np.mod(start_rad, 2*np.pi)
        end_rad = np.mod(end_rad, 2*np.pi)

        if start_rad <= end_rad:
            mask = (theta_wrapped >= start_rad) & (theta_wrapped <= end_rad)
        else:
            mask = (theta_wrapped >= start_rad) | (theta_wrapped <= end_rad)

        theta_sel = theta[mask]
        r_sel = r[mask]
        # -----------------------------
        # Plot directivity
        # -----------------------------
        cx, cy = group_mid

        x_dir = cx + r_sel*np.cos(theta_sel)
        y_dir = cy + r_sel*np.sin(theta_sel)

        ax.plot(x_dir, y_dir, color='red', linewidth=2,
                label='Directivity (Proposed Method)' if i == 0 else None, zorder = 50)
        ax.scatter(cx, cy, c='red', s=60,
                   label='Array center' if i == 0 else None, zorder = 50)

        # -----------------------------
        # Overlay curves
        # -----------------------------
        overlay_colors = ["k", "tab:purple", "tab:brown", "tab:gray"]

        for curve_idx, curve in enumerate(overlay_curve_list):
            curve = np.asarray(curve, dtype=float)

            # Apply offset logic
            if i == 0:
                theta_overlay = np.deg2rad(np.mod(curve[:, 0] + off_set, 360.0))
                db_sel = np.flip(curve[:, 1])
            else:
                theta_overlay = np.deg2rad(np.mod(curve[:, 0], 360.0))
                db_sel = curve[:, 1]

            # Mean alignment
            db_mean = np.mean(db_sel)
            db_aligned = db_sel - db_mean + target_mean_db
            db_clipped = np.maximum(db_aligned, directivity_floor_db)

            r_overlay = directivity_scale * (db_clipped - directivity_floor_db) / (0 - directivity_floor_db)

            cx, cy = group_mid
            x_overlay = cx + r_overlay * np.cos(theta_overlay)
            y_overlay = cy + r_overlay * np.sin(theta_overlay)

            ax.plot(
                x_overlay,
                y_overlay,
                linestyle='none',
                marker='o',
                markersize=3,
                color=overlay_colors[curve_idx % len(overlay_colors)],
                label=overlay_labels[curve_idx] if (overlay_labels and i == 0) else None,
                zorder=5,
            )

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xticks(fontsize = 20)
    ax.set_yticks(fontsize = 20)
    ax.set_xlabel("X (m)", fontsize = 20)
    ax.set_ylabel("Y (m)", fontsize = 20)

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    fig.tight_layout()

    if show:
        plt.show()

    return fig

# def plot_arrows_with_directivity(
#     est_filters,
#     spk_pos,
#     lstn_pos,
#     loc,
#     fs,
#     group_idx,
#     unc=None,
#     col=None,
#     target_pos=None,
#     vbap_values=None,
#     show=True,
#     n_angles=360,
#     freqs_eval_hz=(250, 500, 1000, 2000, 4000),
#     c=343.0,
#     directivity_mode="avg",   # "avg" or "single"
#     directivity_freq_idx=2,   # used if mode="single"
#     directivity_scale=0.3,
#     directivity_floor_db=-20,
#     overlay_curve=None,
#     overlay_curves=None,
#     overlay_labels=None,
# ):
#     # -----------------------------
#     # Prepare inputs
#     # -----------------------------
#     if unc is None:
#         unc = torch.zeros_like(loc)
#     if col is None:
#         col = torch.zeros_like(loc)

#     lstn_pos = lstn_pos.detach().cpu().numpy().squeeze().reshape(-1, 2)
#     spk_pos = spk_pos.detach().cpu().numpy().squeeze()
#     loc = loc.detach().cpu().numpy().squeeze().reshape(-1)
#     unc = unc.detach().cpu().numpy().squeeze().reshape(-1)
#     col = col.detach().cpu().numpy().squeeze().reshape(-1)

#     global_avg_levels = []
#     overlay_curve_list = []
#     if overlay_curve is not None:
#         overlay_curve_list.append(overlay_curve)
#     if overlay_curves is not None:
#         overlay_curve_list.extend(overlay_curves)

#     if overlay_labels is not None and len(overlay_labels) != len(overlay_curve_list):
#         raise ValueError("overlay_labels must contain one label per overlay curve.")


#     # -----------------------------
#     # Compute AVERAGED overlay curve
#     # -----------------------------
#     overlay_avg_theta = None
#     overlay_avg_db = None

#     if len(overlay_curve_list) > 0:
#         interp_powers = []

#         # Use theta grid from directivity for consistency
#         theta_grid = theta  # already defined in your loop

#         for curve in overlay_curve_list:
#             curve = np.asarray(curve, dtype=float)

#             angles = np.deg2rad(np.mod(curve[:, 0], 360.0))
#             db_vals = curve[:, 1]

#             # Sort (important for interpolation)
#             sort_idx = np.argsort(angles)
#             angles = angles[sort_idx]
#             db_vals = db_vals[sort_idx]

#             # Convert to linear power
#             power = 10 ** (db_vals / 10.0)

#             # Interpolate onto common grid
#             power_interp = np.interp(theta_grid, angles, power, period=2*np.pi)

#             interp_powers.append(power_interp)

#         # Average in linear domain
#         mean_power = np.mean(np.stack(interp_powers, axis=0), axis=0)

#         # Back to dB
#         overlay_avg_db = 10 * np.log10(mean_power + 1e-12)
#         overlay_avg_theta = theta_grid


#     # -----------------------------
#     # Create main plot
#     # -----------------------------
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Speakers & listeners
#     ax.scatter(spk_pos[:, 0], spk_pos[:, 1], c='black', marker='x', label='Speakers')
#     ax.scatter(lstn_pos[:, 0], lstn_pos[:, 1], c='blue', label='Listeners')

#     # -----------------------------
#     # Arrow plotting
#     # -----------------------------
#     arrow_len = 0.2
#     cmap = cm.viridis
#     norm = colors.Normalize(vmin=0, vmax=18.5)

#     for x, y, ang, leng, col_val in zip(lstn_pos[:, 0], lstn_pos[:, 1], loc, unc, col):
#         dx = (1 - leng) * arrow_len * np.cos(np.deg2rad(ang))
#         dy = (1 - leng) * arrow_len * np.sin(np.deg2rad(ang))
#         color = cmap(norm(max(col_val, 1e-3)))

#         ax.arrow(x, y, dx, dy,
#                  head_width=0.05, head_length=0.08,
#                  fc=color, ec=color)

#     # Colorbar
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     fig.colorbar(sm, ax=ax, label="LSD")

#     # -----------------------------
#     # Compute DIRECTIVITY (same as your other function)
#     # -----------------------------

#     for i,g in enumerate(group_idx):
        
#         group_indices =  g

#         filters = np.atleast_2d(np.asarray(est_filters, dtype=float))
#         group_indices = np.asarray(group_indices, dtype=int).reshape(-1)

#         h_group = filters[group_indices]
#         p_group = spk_pos[group_indices]

#         # -----------------------------
#         # Compute group center
#         # -----------------------------
#         group_mid = np.mean(p_group, axis=0)


#         n_taps = h_group.shape[1]
#         theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

#         n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
#         freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
#         omega = 2 * np.pi * freqs
#         H_group = np.fft.rfft(h_group, n=n_fft, axis=1)

#         freq_bin_idx = [np.argmin(np.abs(freqs - f)) for f in freqs_eval_hz]

#         mag2 = np.zeros((len(freqs_eval_hz), len(theta)))

#         radius = np.mean(np.linalg.norm(p_group - group_mid, axis=1))

#         for k, ang in enumerate(theta):
#             obs = group_mid + np.array([radius * np.cos(ang), radius * np.sin(ang)])
#             tau = np.linalg.norm(obs - p_group, axis=1) / c
#             phase = np.exp(-1j * np.outer(tau, omega))
#             H_sum = np.sum(H_group * phase, axis=0)

#             for fi, bi in enumerate(freq_bin_idx):
#                 mag2[fi, k] = np.abs(H_sum[bi]) ** 2

#         dir_db = 10 * np.log10(mag2 + 1e-12)

#         # -----------------------------
#         # Select which directivity to plot
#         # -----------------------------
#         if directivity_mode == "avg":
#             dir_use = 10 * np.log10(np.mean(mag2, axis=0) + 1e-12)
#         elif directivity_mode == "single":
#             dir_use = dir_db[directivity_freq_idx]
#         else:
#             raise ValueError("directivity_mode must be 'avg' or 'single'")

#         # -----------------------------
#         # Convert dB → radius (normalized)
#         # -----------------------------
#         dir_clipped = np.maximum(dir_use, directivity_floor_db)

#         r = directivity_scale * (dir_clipped - directivity_floor_db) / (0 - directivity_floor_db)

#         curve = np.asarray(overlay_curve_list[0], dtype=float) if overlay_curve_list else None

#         start_angle = np.min(np.mod(curve[:,0], 360.0)) if i == 0 else np.min(np.mod(curve[:,0], 360.0))
#         end_angle = np.max(np.mod(curve[:,0], 360.0)) if i == 0 else np.max(np.mod(curve[:,0], 360.0))

#         if i == 0:
#             off_set = 180-start_angle
#             start_angle += off_set
#             end_angle += off_set 
            
#         # Convert to radians
#         start_rad = np.deg2rad(start_angle)
#         end_rad = np.deg2rad(end_angle)

#         # Wrap to [0, 2π]
#         theta_wrapped = np.mod(theta, 2*np.pi)
#         start_rad = np.mod(start_rad, 2*np.pi)
#         end_rad = np.mod(end_rad, 2*np.pi)

#         # Handle wrap-around (e.g. 300° → 60°)
#         if start_rad <= end_rad:
#             mask = (theta_wrapped >= start_rad) & (theta_wrapped <= end_rad)
#         else:
#             mask = (theta_wrapped >= start_rad) | (theta_wrapped <= end_rad)

#         # Apply mask
#         theta_sel = theta[mask]
#         r_sel = r[mask]


#         # -----------------------------
#         # Plot directivity in Cartesian coords
#         # -----------------------------
#         cx, cy = group_mid

#         x_dir = cx + r_sel * np.cos(theta_sel)
#         y_dir = cy + r_sel * np.sin(theta_sel)

#         ax.plot(x_dir, y_dir, color='black', linewidth=2, zorder=3, label='Directivity' if i == 0 else None)
#         ax.fill(x_dir, y_dir, color='gray', alpha=0.2, zorder=2)

#         # Mark center
#         ax.scatter(cx, cy, c='red', s=60, label='Array center' if i == 0 else None, zorder=4)


#                 # -----------------------------
#         # Overlay curves (with angle mask)
#         # -----------------------------
#         overlay_curve_list = []
#         if overlay_curve is not None:
#             overlay_curve_list.append(overlay_curve)
#         if overlay_curves is not None:
#             overlay_curve_list.extend(overlay_curves)

#         if overlay_labels is not None and len(overlay_labels) != len(overlay_curve_list):
#             raise ValueError("overlay_labels must contain one label per overlay curve.")

#         overlay_colors = ["k", "tab:purple", "tab:brown", "tab:gray", "tab:olive"]


#         cx, cy = group_mid

#         for curve_idx, curve in enumerate(overlay_curve_list):
#             curve = np.asarray(curve, dtype=float)

#             if curve.ndim != 2 or curve.shape[1] != 2:
#                 raise ValueError("Each overlay curve must have shape (N, 2) as [angle_deg, level_db].")

#             theta_sel = np.deg2rad(np.mod(curve[:, 0], 360.0)) if i == 1 else np.deg2rad(np.mod(curve[:, 0], 360.0) + off_set)
#             db_sel = curve[:, 1] if i == 1 else np.flip(curve[:, 1])

#             # --- dB → radius (same scaling) ---
#             db_clipped = np.maximum(db_sel, directivity_floor_db)
#             r_overlay = directivity_scale * (db_clipped - directivity_floor_db) / (0 - directivity_floor_db)

#             # --- Cartesian ---
#             x_overlay = cx + r_overlay * np.cos(theta_sel)
#             y_overlay = cy + r_overlay * np.sin(theta_sel)

#             # --- Plot ---
#             ax.plot(
#                 x_overlay,
#                 y_overlay,
#                 linestyle='none',
#                 marker='o',
#                 markersize=3,
#                 color=overlay_colors[curve_idx % len(overlay_colors)],
#                 label= (
#                         overlay_labels[curve_idx]
#                         if (overlay_labels is not None and i == 0)
#                         else (f"Overlay {curve_idx+1}" if (overlay_labels is None and i == 0) else None)
#                     ),
#                 zorder=5,
#             )

#     # -----------------------------
#     # VBAP arrows (optional)
#     # -----------------------------
#     if vbap_values is not None:
#         loc_vbap, unc_vbap, _ = vbap_values

#         loc_vbap = loc_vbap.detach().cpu().numpy().reshape(-1)
#         unc_vbap = unc_vbap.detach().cpu().numpy().reshape(-1)

#         for x, y, ang, leng in zip(lstn_pos[:, 0], lstn_pos[:, 1], loc_vbap, unc_vbap):
#             dx = (1 - leng) * arrow_len * np.cos(np.deg2rad(ang))
#             dy = (1 - leng) * arrow_len * np.sin(np.deg2rad(ang))

#             ax.arrow(x, y, dx, dy,
#                      head_width=0.05, head_length=0.08,
#                      fc='purple', ec='purple')

#     # -----------------------------
#     # Target position
#     # -----------------------------
#     if target_pos is not None:
#         target_pos = np.asarray(target_pos).reshape(-1)
#         ax.scatter(target_pos[0], target_pos[1],
#                    c='green', marker='x', s=80, label='Target')

#     # -----------------------------
#     # Formatting
#     # -----------------------------
#     ax.set_aspect('equal')
#     ax.grid(True)
#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")

#     handles, labels = ax.get_legend_handles_labels()
#     unique = dict(zip(labels, handles))
#     ax.legend(unique.values(), unique.keys())

#     fig.tight_layout()

#     if show:
#         plt.show()

#     return fig