import torch
import numpy as np
import matplotlib.pyplot as plt

from scripts.perceptual_models.llado2025.llado2025_estimateinterauralcues import llado2025_estimateinterauralcues

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
    n_fft = 4096
    freqs = np.fft.rfftfreq(n_fft, 1/fs)

    mag_left = 20*np.log10(np.abs(np.fft.rfft(est_filters[0], n=n_fft)))
    mag_right = 20*np.log10(np.abs(np.fft.rfft(est_filters[1], n=n_fft)))

    fig_freq, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

    ax[0].semilogx(freqs, mag_left)
    ax[0].set_title('Left FIR (Freq)')
    ax[0].grid()

    ax[1].semilogx(freqs, mag_right)
    ax[1].set_title('Right FIR (Freq)')
    ax[1].grid()

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
        ax.scatter(spk_pos[0], spk_pos[1], c='red', label='Speaker')
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

        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)
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
    
