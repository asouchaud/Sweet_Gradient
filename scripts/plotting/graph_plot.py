import torch
import numpy as np
import matplotlib.pyplot as plt

from scripts.perceptual_models.llado2025.llado2025_estimateinterauralcues import llado2025_estimateinterauralcues


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
    - Magnitude is in dB: \(20 \\log_{10}(|H| + eps)\)
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

def plot_time(est_filters, show = True):
    
    # -------- Time domain --------
    fig_time, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

    ax[0].plot(est_filters[0], label='Left loudspeaker filter')
    ax[0].set_title('Left FIR')
    ax[0].grid()

    ax[1].plot(est_filters[1], label='Right loudspeaker filter')
    ax[1].set_title('Right FIR')
    ax[1].grid()

    fig_time.suptitle('Estimated FIRs after training')
    fig_time.tight_layout()

    if show:
        plt.show()
    
    return(fig_time)

def plot_freq(est_filters, fs, show = True):

    # -------- Frequency domain --------
    fig_freq, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

    freqs_l, mag_l, _ = rfft_mag_db_phase(est_filters[0], fs, n_fft=4096)
    freqs_r, mag_r, _ = rfft_mag_db_phase(est_filters[1], fs, n_fft=4096)
    plot_mag_db_from_response(freqs_l, mag_l, ax=ax[0], title='Left FIR (Freq)')
    plot_mag_db_from_response(freqs_r, mag_r, ax=ax[1], title='Right FIR (Freq)')

    fig_freq.suptitle('Estimated FIRs (Frequency Domain)')
    fig_freq.tight_layout()

    if show:
        plt.show()

    return fig_freq   

def plot_arrows(lstn_pos, spk_pos, loc, unc = None, col=None, loc_target = None, show = True):
        
        if unc is None:
            unc = torch.zeros_like(loc, device=loc.device, dtype=loc.dtype)
        
        if col is None:
            col = torch.zeros_like(loc, device=loc.device, dtype=loc.dtype)
    
        
        lstn_pos = lstn_pos.detach().cpu().numpy().squeeze().reshape(-1, 2) # WORKS ONLY FOR BATCH SIZE 1
        spk_pos = spk_pos.detach().cpu().numpy().squeeze()
        loc = loc.detach().cpu().numpy().squeeze().reshape(-1)
        unc = unc.detach().cpu().numpy().squeeze().reshape(-1)
        col = col.detach().cpu().numpy().squeeze().reshape(-1)

        fig_arrow, ax = plt.subplots(figsize=(6,6))
        ax.scatter(spk_pos[:,0], spk_pos[:,1], c='red', label='Speaker')
        ax.scatter(lstn_pos[:,0], lstn_pos[:,1], c='blue', label='Listener positions')

        if loc_target is not None:
            loc_pos = [1, np.tan(np.deg2rad(loc_target.item()))]
            ax.scatter(loc_pos[0], loc_pos[1], c='green', label='Target position')

        
        arrow_len = 0.2
        for x, y, ang, unc in zip(lstn_pos[:,0], lstn_pos[:,1], loc, unc):  

            dx = (1-unc)*arrow_len * np.cos(np.deg2rad(ang))
            dy = (1-unc)*arrow_len * np.sin(np.deg2rad(ang))

            ax.arrow(float(x), float(y), float(dx), float(dy),
                    head_width=0.05, head_length=0.08, fc='orange', ec='orange')    

        ax.set_ylim(-1.25,1.25)
        ax.set_xlim(-0.5,2.0)
        ax.grid()
        ax.set_ylabel('Y position (m)')
        ax.set_xlabel('X position (m)')
        ax.legend()

        if show:
            plt.show()
        
        return fig_arrow

def plot_bin_cues(bin_stim, fs, gtfb, show = True):

    interaural_cues = llado2025_estimateinterauralcues(bin_stim=bin_stim, fs = fs, gtfb=gtfb)
    
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

def plot_color(bin_stim, ref_fft, lstn_pos, show = True):

    bin_stim_fft = 20*np.log(np.abs(np.fft.rfft(bin_stim, n=ref_fft.shape[-1]*2-1, axis = -1)))

    ref_fft = ref_fft.detach().cpu().numpy()
    if ref_fft.shape[0] == 1:
        ref_fft = np.repeat(ref_fft, bin_stim_fft.shape[0], axis=0)

    freq = np.fft.rfftfreq(ref_fft.shape[-1], 1/48000)
   
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
        axes_l[i].semilogx(bin_stim_fft[i,0], label = "Estimated Signal")  
        axes_l[i].semilogx(ref_fft[i,0], label = "Reference Signal")
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
        axes_r[i].semilogx(bin_stim_fft[i,1], label = "Estimated Signal")  
        axes_r[i].semilogx(ref_fft[i,1], label = "Reference Signal")
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
    
    
