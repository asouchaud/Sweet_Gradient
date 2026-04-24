import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from scripts.helpers.peripheral_processing import estimate_interauralcues


def rfft_freq_response(h, fs, n_fft: int = 4096):
    """
    Shared helper for FIR frequency response used by both `plot_freq` and video rendering.

    Parameters
    ----------
    h : array-like
        FIR(s) with shape (..., n_taps).
    fs : float
        Sampling rate in Hz.
    n_fft : int
        FFT size (mirrors `plot_freq` default).

    Returns
    -------
    freqs_hz : np.ndarray
        Frequency axis in Hz for the rFFT bins.
    H : np.ndarray (complex)
        rFFT of `h` along the last axis, shape (..., n_bins).
    """
    h = np.asarray(h)
    H = np.fft.rfft(h, n=n_fft, axis=-1)
    freqs_hz = np.fft.rfftfreq(n_fft, 1 / float(fs))
    return freqs_hz, H


def rfft_mag_db_phase(h, fs, n_fft: int = 4096, eps: float = 1e-12):
    """
    Compute rFFT magnitude (in dB) and phase (in radians).

    Notes
    -----
    - Magnitude is in dB: 20 * log10(|H| + eps).
    - Phase is in radians: angle(H) in [-pi, pi]
    """
    freqs_hz, H = rfft_freq_response(h, fs, n_fft=n_fft)
    mag_db = 20.0 * np.log10(np.abs(H) + float(eps))
    phase = np.angle(H)
    return freqs_hz, mag_db, phase


def plot_mag_db_from_response(
    freqs_hz,
    mag_db,
    ax=None,
    title: str = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Magnitude (dB)",
    return_line: bool = False,
):
    """
    Plot magnitude response in dB on a log-frequency x-axis.

    This is the common routine used by `plot_freq` and the training GIF.
    It masks out DC/non-positive freqs (required for log-x), sets x-limits,
    labels, and grid styling.
    """
    freqs_hz = np.asarray(freqs_hz).reshape(-1)
    mag_db = np.asarray(mag_db).reshape(-1)
    if freqs_hz.shape != mag_db.shape:
        raise ValueError(f"freqs_hz and mag_db must have same shape, got {freqs_hz.shape} vs {mag_db.shape}")

    mask = np.isfinite(freqs_hz) & np.isfinite(mag_db) & (freqs_hz > 0.0)
    f_plot = freqs_hz[mask]
    mag_plot = mag_db[mask]
    if f_plot.size == 0:
        raise ValueError("No valid positive-frequency points available to plot on log-x axis.")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
    else:
        fig = ax.figure

    (line,) = ax.semilogx(f_plot, mag_plot)
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(float(f_plot[0]), float(f_plot[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    if return_line:
        return fig, ax, line, f_plot
    return fig, ax


def _stem_line_segments(x_axis, y_vals, bottom = 0.0):
    return [
        np.array([[float(xi), float(bottom)], [float(xi), float(yi)]], dtype=float)
        for xi, yi in zip(x_axis, y_vals)
    ]


def _apply_axis_limits(ax, limits, fallback = None, symmetric = False):
    if limits is None:
        limits = fallback

    if limits is None:
        return

    if np.isscalar(limits):
        limit = float(limits)
        if symmetric:
            ax.set_ylim(-abs(limit), abs(limit))
        else:
            if fallback is None:
                current_min, _ = ax.get_ylim()
                ax.set_ylim(current_min, limit)
            else:
                ax.set_ylim(float(fallback[0]), limit)
        return

    if len(limits) != 2:
        raise ValueError("Axis limits must be a scalar or a (min, max) pair.")

    ax.set_ylim(float(limits[0]), float(limits[1]))


def plot_stem_signal(
    y, 
    ax = None, 
    title = None, 
    max_min = None, 
    xlabel = "Sample",
    ylabel = "Amplitude", 
    label = None, 
    bottom = 0.0, 
    return_artists = False):
    """
    Reusable stem-plot helper for FIR filters or other 1-D sequences.

    `max_min` can be a scalar (used as symmetric ±limit) or a `(min, max)` pair.
    """
    y = np.asarray(y).reshape(-1)
    x = np.arange(y.size, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
    else:
        fig = ax.figure

    stem = ax.stem(x, y, linefmt="-", markerfmt="o", basefmt=" ", bottom=bottom, label=label)

    if x.size > 1:
        ax.set_xlim(float(x[0]), float(x[-1]))
    elif x.size == 1:
        ax.set_xlim(float(x[0]) - 0.5, float(x[0]) + 0.5)
    else:
        ax.set_xlim(0.0, 1.0)

    _apply_axis_limits(ax, max_min, symmetric=True)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    if label is not None:
        ax.legend()

    if return_artists:
        return fig, ax, stem, x
    return fig, ax


def update_stem_artists(stem, x_axis, y_vals, bottom = 0.0):
    x_axis = np.asarray(x_axis).reshape(-1)
    y_vals = np.asarray(y_vals).reshape(-1)

    stem.markerline.set_xdata(x_axis)
    stem.markerline.set_ydata(y_vals)

    if hasattr(stem.stemlines, "set_segments"):
        stem.stemlines.set_segments(_stem_line_segments(x_axis, y_vals, bottom=bottom))
    else:
        for line, xi, yi in zip(stem.stemlines, x_axis, y_vals):
            line.set_data([float(xi), float(xi)], [float(bottom), float(yi)])

    return stem


def plot_frequency_response(
    h = None,
    fs = None,
    freqs_hz = None,
    mag_db = None,
    phase = None,
    ax_mag = None,
    ax_phase = None,
    n_fft: int = 4096,
    mag_title = None,
    phase_title = "Phase",
    mag_max = None,
    include_phase = False,
    return_lines = False,
):
    """
    Reusable frequency-response helper.

    Either provide `h` and `fs`, or precomputed `freqs_hz` and `mag_db`
    (plus `phase` when `include_phase=True`).

    `mag_max` can be a scalar (used as the upper y-limit) or a `(min, max)` pair.
    """
    if freqs_hz is None or mag_db is None or (include_phase and phase is None):
        if h is None or fs is None:
            raise ValueError("Either provide h and fs, or precomputed freqs_hz/mag_db/(phase).")
        freqs_hz, mag_db, phase = rfft_mag_db_phase(h, fs, n_fft=n_fft)

    freqs_hz = np.asarray(freqs_hz).reshape(-1)
    mag_db = np.asarray(mag_db).reshape(-1)
    phase_array = None if phase is None else np.asarray(phase).reshape(-1)

    if include_phase and phase_array is None:
        raise ValueError("phase must be provided when include_phase=True.")

    if ax_mag is None:
        if include_phase:
            fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(10, 2.5))
        else:
            fig, ax_mag = plt.subplots(1, 1, figsize=(10, 2.5))
            ax_phase = None
    else:
        fig = ax_mag.figure
        if include_phase and ax_phase is None:
            raise ValueError("ax_phase must be provided when include_phase=True and ax_mag is supplied.")

    fig, ax_mag, line_mag, f_plot = plot_mag_db_from_response(
        freqs_hz,
        mag_db,
        ax=ax_mag,
        title=mag_title,
        return_line=True,
    )

    if mag_max is not None:
        mag_mask = np.isfinite(freqs_hz) & np.isfinite(mag_db) & (freqs_hz > 0.0)
        mag_plot = mag_db[mag_mask]
        if mag_plot.size == 0:
            raise ValueError("No valid positive-frequency points available to plot.")

        _apply_axis_limits(
            ax_mag,
            mag_max,
            fallback=(float(np.min(mag_plot)), float(np.max(mag_plot))),
            symmetric=False,
        )

    line_phase = None
    phase_plot = None

    if include_phase:
        phase_mask = np.isfinite(freqs_hz) & np.isfinite(phase_array) & (freqs_hz > 0.0)
        f_phase = freqs_hz[phase_mask]
        phase_plot = phase_array[phase_mask]

        if f_phase.size == 0:
            raise ValueError("No valid positive-frequency points available for the phase plot.")

        (line_phase,) = ax_phase.plot(f_phase, phase_plot, lw=1.2)
        ax_phase.set_xscale("log")
        ax_phase.set_xlim(float(f_phase[0]), float(f_phase[-1]))
        ax_phase.set_ylim(-np.pi, np.pi)
        ax_phase.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
        ax_phase.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylabel("Phase (rad)")
        if phase_title is not None:
            ax_phase.set_title(phase_title)
        ax_phase.grid(True, alpha=0.25)

    if return_lines:
        return fig, ax_mag, ax_phase, line_mag, line_phase, f_plot, phase_plot
    return fig, ax_mag, ax_phase


def plot_time(est_filters, show = True):
    
    est_filters = np.atleast_2d(np.asarray(est_filters))
    n_sources = est_filters.shape[0]
    max_min = float(np.max(np.abs(est_filters))) if est_filters.size else 1.0
    if max_min == 0.0:
        max_min = 1.0

    # -------- Time domain --------
    fig_time, ax = plt.subplots(n_sources, 1, sharex=True, sharey=True, figsize=(10, 2.5 * n_sources))
    ax = np.atleast_1d(ax)

    for source_idx in range(n_sources):
        plot_stem_signal(
            est_filters[source_idx],
            ax=ax[source_idx],
            title=f"Source {source_idx + 1} FIR",
            max_min=max_min,
        )

    fig_time.suptitle('Estimated FIRs after training')
    fig_time.tight_layout()

    if show:
        plt.show()
    
    return(fig_time)

def plot_freq(est_filters, fs, show = True):

    est_filters = np.atleast_2d(np.asarray(est_filters))
    n_sources = est_filters.shape[0]

    # -------- Frequency domain --------
    fig_freq, ax = plt.subplots(n_sources, 1, sharex=True, sharey=True, figsize=(10, 2.5 * n_sources))
    ax = np.atleast_1d(ax)

    for source_idx in range(n_sources):
        plot_frequency_response(
            h=est_filters[source_idx],
            fs=fs,
            ax_mag=ax[source_idx],
            mag_title=f'Source {source_idx + 1} FIR (Freq)',
        )

    fig_freq.suptitle('Estimated FIRs (Frequency Domain)')
    fig_freq.tight_layout()

    if show:
        plt.show()

    return fig_freq   

def plot_arrows(lstn_pos, spk_pos, loc, unc=None, col=None, target_pos=None, show=True):

    assert isinstance(lstn_pos, torch.Tensor), "lstn_pos should be a torch.Tensor of shape (n_listeners, 2)"
    assert isinstance(spk_pos, torch.Tensor), "spk_pos should be a torch.Tensor of shape (n_sources, 2)"
    assert isinstance(loc, torch.Tensor), "loc should be a torch.Tensor of shape (n_listeners,)"

    if unc is None:
        unc = torch.zeros_like(loc, device=loc.device, dtype=loc.dtype)
    else:
        assert isinstance(unc, torch.Tensor), "unc should be a torch.Tensor of shape (n_listeners,)"
        assert unc.shape == loc.shape, "unc should have the same shape as loc"
    
    if col is None:
        col = torch.zeros_like(loc, device=loc.device, dtype=loc.dtype)
    else:
        assert isinstance(col, torch.Tensor), "col should be a torch.Tensor of shape (n_listeners,)"
        assert col.shape == loc.shape, "col should have the same shape as loc"

    # Convert to numpy
    lstn_pos = lstn_pos.detach().cpu().numpy().squeeze().reshape(-1, 2) 
    spk_pos = spk_pos.detach().cpu().numpy().squeeze()
    loc = loc.detach().cpu().numpy().squeeze().reshape(-1)
    unc = unc.detach().cpu().numpy().squeeze().reshape(-1)
    col = col.detach().cpu().numpy().squeeze().reshape(-1)

    fig_arrow, ax = plt.subplots(figsize=(6,6))
    ax.scatter(spk_pos[:,0], spk_pos[:,1], c='red', label='Speaker')
    ax.scatter(lstn_pos[:,0], lstn_pos[:,1], c='blue', label='Listener positions')

    arrow_len = 0.2

    # Create a colormap with log scaling
    max_col = 18.5
    norm = colors.LogNorm(vmin=1e-3, vmax=max_col)  # avoid zero
    cmap = cm.viridis

    for x, y, ang, unc_val, col_val in zip(lstn_pos[:,0], lstn_pos[:,1], loc, unc, col):
        dx = (1-unc_val)*arrow_len * np.cos(np.deg2rad(ang))
        dy = (1-unc_val)*arrow_len * np.sin(np.deg2rad(ang))

        color = cmap(norm(max(col_val, 1e-3)))  # avoid zero for log scale

        ax.arrow(float(x), float(y), float(dx), float(dy),
                 head_width=0.05, head_length=0.08, fc=color, ec=color)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_arrow.colorbar(sm, ax=ax)
    cbar.set_label("Colouration / Loss")

    x_min = float(np.min(spk_pos[:,0]))
    x_max = float(np.max(spk_pos[:,0]))
    y_min = float(np.min(spk_pos[:,1]))
    y_max = float(np.max(spk_pos[:,1]))
    x_pad = max(0.1, 0.1 * max(x_max - x_min, 1e-9))
    y_pad = max(0.1, 0.1 * max(y_max - y_min, 1e-9))
    current_x_min = float(np.min(lstn_pos[:,0]))
    current_x_max = float(np.max(lstn_pos[:,0]))
    current_y_min = float(np.min(lstn_pos[:,1]))
    current_y_max = float(np.max(lstn_pos[:,1]))
    ax.set_xlim(min(current_x_min, x_min - x_pad), max(current_x_max, x_max + x_pad))
    ax.set_ylim(min(current_y_min, y_min - y_pad), max(current_y_max, y_max + y_pad))
    ax.autoscale(enable=False)

    if target_pos is not None:
        target_pos_np = None
        if torch.is_tensor(target_pos):
            target_pos_np = target_pos.detach().cpu().numpy().reshape(-1)
        else:
            target_pos_np = np.asarray(target_pos, dtype=float).reshape(-1)

        ax.scatter(target_pos_np[0], target_pos_np[1], c='green', marker='X', label='Target position')

    ax.grid()
    ax.set_ylabel('Y position (m)')
    ax.set_xlabel('X position (m)')
    ax.legend()

    if show:
        plt.show()

    return fig_arrow


def plot_bin_cues(bin_stim, fs, gtfb, show = True):

    if len(bin_stim.shape) == 2:
        bin_stim = bin_stim.unsqueeze(0).unsqueeze(0)
    elif len(bin_stim.shape) == 3:
        bin_stim = bin_stim.unsqueeze(0)
    interaural_cues = estimate_interauralcues(bin_stim=bin_stim, fs = fs, gtfb=gtfb)
    
    ild_np = interaural_cues['ild'].detach().cpu().numpy()
    itd_np = interaural_cues['itd'].detach().cpu().numpy()

    colors = np.arange(len(ild_np[0,0]))
    
    n_plots = ild_np.shape[1]

    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots / cols))

    fig_bin_cues, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))

    axes = np.array(axes).flatten() if n_plots > 1 else np.array([axes])
    fig_bin_cues.tight_layout(rect=[0, 0, 1, 0.95]) 
    for i, ax in enumerate(axes):
        if i < n_plots:
            sc = ax.scatter(
                itd_np[0,i] * 1e3,
                ild_np[0,i],
                c=colors,
                cmap='viridis',
            )

            if i % cols == cols-1:
                fig_bin_cues.colorbar(sc, ax=ax, label='Gammatone filter bank index')
            ax.set_title(f'Listener {i+1}')
            if i % cols == 0:
                ax.set_ylabel('ILD [dB]')
            if i // cols == rows-1:
                ax.set_xlabel('ITD [ms]')
            ax.set_ylim(-20, 20)
            ax.set_xlim(-1.5, 1.5)
            ax.grid(True)
        else:
            ax.set_visible(False)

    if show:
        plt.show()

    return fig_bin_cues

def plot_color(bin_stim, ref_stim, lstn_pos, show = True):

    ref_stim_np = ref_stim.detach().cpu().numpy()
    n_fft = 4096

    bin_stim_fft = 20*np.log(np.abs(np.fft.rfft(bin_stim, n=n_fft, axis = -1)))
    ref_fft = 20*np.log(np.abs(np.fft.rfft(ref_stim_np, n=n_fft, axis = -1)))

    if len(bin_stim_fft.shape) == 4 and len(ref_fft.shape) == 4:
        bin_stim_fft = bin_stim_fft.mean(axis=0)
        ref_fft = ref_fft.mean(axis=0)

    if len(bin_stim_fft.shape) == 2:
        bin_stim_fft = bin_stim_fft[None,:,:]
    if ref_fft.shape[0] == 1:
        ref_fft = np.repeat(ref_fft, bin_stim_fft.shape[0], axis=0)

    freq = np.fft.rfftfreq(n_fft, 1/48000)
   
    lstn_pos = lstn_pos.detach().cpu().numpy().squeeze().reshape(-1, 2)

    n_pos = lstn_pos.shape[0]

    # Compute number of rows and columns
    cols = np.ceil(np.sqrt(n_pos)).astype(int)      # try to make square-ish
    rows = np.ceil(n_pos / cols).astype(int)  

    # Create subplots
    fig_l, axes_l = plt.subplots(rows, cols, figsize=(cols*5, rows*3))

    ## Left ear

    # Flatten axes array for easy indexing (in case of multiple rows/cols)
    axes_l = axes_l.flatten() if n_pos > 1 else [axes_l]

    for i in range(n_pos):
        axes_l[i].semilogx(freq[9:], bin_stim_fft[i,0,9:], label = "Estimated Signal")  
        axes_l[i].semilogx(freq[9:], ref_fft[i,0,9:], label = "Reference Signal")
        axes_l[i].set_title(f"Position {lstn_pos[i]}")
        axes_l[i].legend()
        axes_l[i].set_ylabel('Magnitude [dB]')
        axes_l[i].set_xlabel('Frequency [Hz]')
        axes_l[i].grid()

    # Hide any unused subplots
    for j in range(n_pos, len(axes_l)):
        axes_l[j].axis('off')

    plt.tight_layout()

    ## Right ear

    # Create subplots
    fig_r, axes_r = plt.subplots(rows, cols, figsize=(cols*5, rows*3))

    # Flatten axes array for easy indexing (in case of multiple rows/cols)
    axes_r = axes_r.flatten() if n_pos > 1 else [axes_r]

    for i in range(n_pos):
        axes_r[i].semilogx(freq[9:], bin_stim_fft[i,1,9:], label = "Estimated Signal")  
        axes_r[i].semilogx(freq[9:], ref_fft[i,1,9:], label = "Reference Signal")
        axes_r[i].set_title(f"Position {lstn_pos[i]}")
        axes_r[i].legend()
        axes_r[i].set_ylabel('Magnitude [dB]')
        axes_r[i].set_xlabel('Frequency [Hz]')#
        axes_r[i].grid()


    # Hide any unused subplots
    for j in range(n_pos, len(axes_r)):
        axes_r[j].axis('off')

    plt.tight_layout()

    if show:
        plt.show()

    return(fig_l, fig_r)
