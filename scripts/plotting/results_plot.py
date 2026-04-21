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

from scripts.helpers.dsp import compute_reference_bin


def plot_results(estimation_dict, est_filters, bin_stim, lstn_pos, spk_pos, loss_dict, target_pos, fs, stim, device, dtype, show=True):
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

    target_marker = torch.as_tensor(target_pos, device=device, dtype=dtype)

    arrow_plot = None
    loc_model = perceptual_models.get("localisation")
    if "localisation" in estimation_dict and loc_model is not None:
        arrow_plot = plot_arrows(
            lstn_pos,
            spk_pos,
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
        
        col_reference = compute_reference_bin(stim, 
                                              col_model.reference_hrirs, 
                                              listener_indices=torch.arange(lstn_pos.shape[0], device=device)).view(stim.shape[0], 
                                                                                                                    col_model.reference_hrirs.shape[0], 
                                                                                                                    2, 
                                                                                                                    -1)
        
    
        color_l_plot, color_r_plot = plot_color(
            bin_stim,
            col_reference,
            lstn_pos,
            show=False,
        )

    if show:
        plt.show()

    return {
        "time_plot": time_plot,
        "freq_plot": freq_plot,
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


def _db_at_nearest_angle_deg(sample_angles_deg, levels_db, target_angle_deg):
    sample_angles_deg = np.asarray(sample_angles_deg, dtype=np.float64).ravel()
    levels_db = np.asarray(levels_db, dtype=np.float64).ravel()
    target_wrapped = float(np.mod(target_angle_deg, 360.0))
    samples_wrapped = np.mod(sample_angles_deg, 360.0)
    separation_deg = np.abs((samples_wrapped - target_wrapped + 180.0) % 360.0 - 180.0)
    nearest_idx = int(np.argmin(separation_deg))
    return float(levels_db[nearest_idx])


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
    show_listener_target_lines=False,
    vbap_values=None,
    show=True,
    n_angles=360,
    freqs_eval_hz=(250, 500, 1000, 2000, 4000),
    directivity_observation_radius_m=None,
    c=343.0,
    directivity_mode="avg",
    directivity_freq_idx=2,
    directivity_scale=1.0,
    directivity_floor_db=-20,
    directivity_label="Proposed Method",
    overlay_curve=None,
    overlay_curves=None,
    overlay_labels=None,
    legend_ncol=None,
    legend_bbox_to_anchor=(0.5, 1.02),
    figsize=(6, 6),
):
    # -----------------------------
    # Prepare inputs
    # -----------------------------
    use_lsd_colorbar = col is not None
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
    # Create plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(spk_pos[:, 0], spk_pos[:, 1], c='black', marker='x', label='Loudspeakers')
    ax.scatter(lstn_pos[:, 0], lstn_pos[:, 1], c='blue', label='Virtual Listeners')

    target_pos_np = None
    if target_pos is not None:
        target_pos_np = np.asarray(target_pos, dtype=float).reshape(-1)
        if target_pos_np.size != 2:
            raise ValueError("target_pos must be a 2D position.")
        ax.scatter(
            target_pos_np[0],
            target_pos_np[1],
            marker="^",
            c="red",
            s=60,
            label="Target",
            zorder=6,
        )

    # -----------------------------
    # Arrows
    # -----------------------------
    arrow_len = 0.2
    cmap = cm.viridis
    norm = colors.Normalize(vmin=0, vmax=18.5)
    arrow_mono_color = "#2E0854"

    for x, y, ang, leng, col_val in zip(lstn_pos[:, 0], lstn_pos[:, 1], loc, unc, col):
        dx = (1 - leng) * arrow_len * np.cos(np.deg2rad(ang))
        dy = (1 - leng) * arrow_len * np.sin(np.deg2rad(ang))
        if use_lsd_colorbar:
            color = cmap(norm(max(col_val, 1e-3)))
        else:
            color = arrow_mono_color

        ax.arrow(x, y, dx, dy,
                 head_width=0.05, head_length=0.08,
                 fc=color, ec=color)

    if use_lsd_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label="LSD")

    if len(overlay_curve_list) > 0:
        curve_ref_global = np.asarray(overlay_curve_list[0])
        overlay_angle_offset_deg = 180.0 - float(np.min(np.mod(curve_ref_global[:, 0], 360.0)))
    else:
        overlay_angle_offset_deg = 0.0

    # -----------------------------
    # Directivity loop
    # -----------------------------
    for i, g in enumerate(group_idx):

        group_indices = np.asarray(g, dtype=int).reshape(-1)

        filters = np.atleast_2d(np.asarray(est_filters, dtype=float))
        h_group = filters[group_indices]
        p_group = spk_pos[group_indices]

        group_mid = np.mean(p_group, axis=0)
        cy = float(group_mid[1])
        if cy > 0.0:
            theta_wedge0 = np.pi
            theta_wedge1 = np.pi + np.pi / 3.0
        elif cy < 0.0:
            theta_wedge0 = np.pi - np.pi / 3.0
            theta_wedge1 = np.pi
        else:
            theta_wedge0 = np.pi
            theta_wedge1 = np.pi + np.pi / 3.0

        theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

        n_taps = h_group.shape[1]
        n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
        omega = 2*np.pi*freqs
        H_group = np.fft.rfft(h_group, n=n_fft, axis=1)

        freq_bin_idx = [np.argmin(np.abs(freqs - f)) for f in freqs_eval_hz]

        mag2 = np.zeros((len(freqs_eval_hz), len(theta)))

        if directivity_observation_radius_m is None:
            radius = float(np.mean(np.linalg.norm(p_group - group_mid, axis=1)))
        else:
            radius = float(directivity_observation_radius_m)

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

        theta_wrapped = np.mod(theta, 2 * np.pi)
        mask = (theta_wrapped >= theta_wedge0) & (theta_wrapped <= theta_wedge1)
        mid_rad = 0.5 * (theta_wedge0 + theta_wedge1)

        delta_theta = np.abs((theta - mid_rad + np.pi) % (2 * np.pi) - np.pi)
        k_mid = int(np.argmin(delta_theta))
        dir_at_mid = float(dir_use[k_mid])

        dir_aligned = dir_use - dir_at_mid
        dir_clipped = np.maximum(dir_aligned, directivity_floor_db)
        r = directivity_scale * (dir_clipped - directivity_floor_db) / (0 - directivity_floor_db)

        theta_sel = theta[mask]
        r_sel = r[mask]
        # -----------------------------
        # Plot directivity
        # -----------------------------
        cx, cy = group_mid

        x_dir = cx + r_sel*np.cos(theta_sel)
        y_dir = cy + r_sel*np.sin(theta_sel)

        ax.plot(x_dir, y_dir, color="tab:blue", linewidth=2,
                label=directivity_label if i == 0 else None, zorder = 50)

        line_extent = 0.0
        for db_ref in (-10.0, 0.0, 10.0):
            line_extent = max(
                line_extent,
                directivity_scale * (db_ref - directivity_floor_db) / (0.0 - directivity_floor_db),
            )
        th_oblique_ray = theta_wedge1 if cy >= 0.0 else theta_wedge0
        n_arc = 72
        th_arc = np.linspace(theta_wedge0, theta_wedge1, n_arc)
        theta_label = 0.5 * (theta_wedge0 + theta_wedge1)
        cos_tl, sin_tl = np.cos(theta_label), np.sin(theta_label)
        label_radial_nudge = 0.04 * line_extent

        ref_line_kw = dict(color="0.45", linewidth=0.9, linestyle="-", zorder=35, alpha=0.75)
        ref_arc_kw = dict(color="0.45", linewidth=0.8, linestyle="--", zorder=34, alpha=0.75)

        ax.plot(
            [cx, cx + line_extent * np.cos(np.pi)],
            [cy, cy + line_extent * np.sin(np.pi)],
            **ref_line_kw,
        )
        ax.plot(
            [cx, cx + line_extent * np.cos(th_oblique_ray)],
            [cy, cy + line_extent * np.sin(th_oblique_ray)],
            **ref_line_kw,
        )
        for db_ref in (-10.0, 0.0, 10.0):
            circle_r = directivity_scale * (db_ref - directivity_floor_db) / (0.0 - directivity_floor_db)
            if circle_r > 0.0:
                ax.plot(
                    cx + circle_r * np.cos(th_arc),
                    cy + circle_r * np.sin(th_arc),
                    **ref_arc_kw,
                )
                if db_ref > 0.0:
                    db_lbl = f"+{int(round(db_ref))} dB"
                elif db_ref < 0.0:
                    db_lbl = f"{int(round(db_ref))} dB"
                else:
                    db_lbl = "0 dB"
                r_lbl = circle_r + label_radial_nudge
                ax.text(
                    cx + r_lbl * cos_tl,
                    cy + r_lbl * sin_tl,
                    db_lbl,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.35",
                    zorder=36,
                )

        mid_angle_deg = float(np.rad2deg(mid_rad) % 360.0)

        # -----------------------------
        # Overlay curves
        # -----------------------------
        overlay_colors = ["k", "tab:purple", "tab:brown", "tab:gray"]

        for curve_idx, curve in enumerate(overlay_curve_list):
            curve = np.asarray(curve, dtype=float)

            angles_deg_plot = np.mod(curve[:, 0] + overlay_angle_offset_deg, 360.0)
            db_sel = np.flip(curve[:, 1])
            if cy < 0.0:
                angles_deg_plot = np.mod(360.0 - angles_deg_plot, 360.0)
            theta_overlay = np.deg2rad(angles_deg_plot)

            db_at_mid = _db_at_nearest_angle_deg(angles_deg_plot, db_sel, mid_angle_deg)
            db_aligned = db_sel - db_at_mid
            db_clipped = np.maximum(db_aligned, directivity_floor_db)

            r_overlay = directivity_scale * (db_clipped - directivity_floor_db) / (0 - directivity_floor_db)

            cx, cy = group_mid
            x_overlay = cx + r_overlay * np.cos(theta_overlay)
            y_overlay = cy + r_overlay * np.sin(theta_overlay)
            order = np.argsort(angles_deg_plot)
            oc = overlay_colors[curve_idx % len(overlay_colors)]
            ax.plot(
                x_overlay[order],
                y_overlay[order],
                linestyle="-",
                linewidth=1.4,
                color=oc,
                label=overlay_labels[curve_idx] if (overlay_labels and i == 0) else None,
                zorder=5,
            )

    if show_listener_target_lines and target_pos_np is not None:
        xlim_before = ax.get_xlim()
        ylim_before = ax.get_ylim()
        tx, ty = float(target_pos_np[0]), float(target_pos_np[1])
        for x, y in zip(lstn_pos[:, 0], lstn_pos[:, 1]):
            p_lstn = np.array([float(x), float(y)], dtype=float)
            p_tgt = np.array([tx, ty], dtype=float)
            d = p_tgt - p_lstn
            dist = float(np.linalg.norm(d))
            if dist < 1e-12:
                continue
            u = d / dist
            extend = 0.2 * dist
            p0 = p_lstn - extend * u
            p1 = p_tgt + extend * u
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                linestyle="--",
                linewidth=0.8,
                color="0.65",
                alpha=0.6,
                zorder=1,
            )
        ax.set_xlim(xlim_before)
        ax.set_ylim(ylim_before)

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_aspect('equal')
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("X (m)", fontsize = 12)
    ax.set_ylabel("Y (m)", fontsize = 12)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if legend_ncol is None:
        legend_ncol = 1 if len(unique) <= 2 else 2
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(pad=0.35, rect=[0.02, 0.02, 0.98, 0.91])

    if show:
        plt.show()

    return fig
