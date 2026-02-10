import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from scripts.perceptual_models.llado2025.llado2025_estimateinterauralcues import llado2025_estimateinterauralcues
from scripts.setup.setup import load_gtfb

class TensorboardLogger:
    def __init__(self, log_dir="runs/train", h_log_interval=50, fs = 48000):
        self.fs = fs
        self.writer = SummaryWriter(log_dir=log_dir)
        self.h_log_interval = h_log_interval

        self.gtfb = load_gtfb(fs, 'cuda', torch.float32)


    def log_losses(self, loss_dict, full_loss, epoch):
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"loss/{key}", value.item(), epoch)
        self.writer.add_scalar("loss/total", full_loss.item(), epoch)

    def log_h_full(self, h_l, h_r, h_full, epoch):
        if epoch % self.h_log_interval != 0:
            return

        # Histograms
        self.writer.add_histogram("h_full/left", h_l.detach(), epoch)
        self.writer.add_histogram("h_full/right", h_r.detach(), epoch)

        h = h_full.squeeze(0).detach().cpu().numpy()

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        ax[0].set_ylim(-2,2)
        ax[1].set_ylim(-2,2)
        ax[0].grid()
        ax[1].grid()
        ax[0].plot(h[0])
        ax[0].set_title("Left FIR")
        ax[1].plot(h[1])
        ax[1].set_title("Right FIR")

        self.writer.add_figure("h_full (Time Domain)", fig, epoch)
        plt.close(fig)
        
        n_fft = 4096
        freqs = np.fft.rfftfreq(n_fft, 1/self.fs)
        h_ft = 20*np.log10(np.abs(np.fft.rfft(h, n = n_fft, axis= -1)))

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        ax[0].set_ylim(-40,40)
        ax[1].set_ylim(-40,40)
        ax[0].grid()
        ax[1].grid()
        ax[0].semilogx(freqs, h_ft[0])
        ax[0].set_title("Left FIR")
        ax[1].semilogx(freqs, h_ft[1])
        ax[1].set_title("Right FIR")

        self.writer.add_figure("h_full (Frequency Domain)", fig, epoch)
        plt.close(fig)

    def log_bin_cues(self, bin_stim, epoch):
        if epoch % self.h_log_interval != 0:
            return

        interaural_cues = llado2025_estimateinterauralcues(bin_stim=bin_stim, fs = self.fs, gtfb=self.gtfb)
        
        ild_np = interaural_cues['ild'].detach().cpu().numpy()
        itd_np = interaural_cues['itd'].detach().cpu().numpy()

        colors = np.arange(len(ild_np[0,0]))
        
        n_plots = ild_np.shape[1]

        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))

        axes = np.array(axes).flatten() if n_plots > 1 else np.array([axes])
        fig.suptitle(f'ILD/ITD pairs at epoch:{epoch}')
        fig.tight_layout(rect=[0, 0, 1, 0.95]) 
        for i, ax in enumerate(axes):
            if i < n_plots:
                sc = ax.scatter(
                    itd_np[0,i] * 1e3,
                    ild_np[0,i],
                    c=colors,
                    cmap='viridis',
                )

                if i % cols == cols-1:
                    fig.colorbar(sc, ax=ax, label='Gammatone filter bank index')
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

        # now this works
        self.writer.add_figure("ILD_ITD", fig, global_step=epoch)
        plt.close(fig)
        pass

    def close(self):
        self.writer.close()

