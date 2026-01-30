import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    def __init__(self, log_dir="runs/train", h_log_interval=50, fs = 48000):
        self.fs = fs
        self.writer = SummaryWriter(log_dir=log_dir)
        self.h_log_interval = h_log_interval

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

    def close(self):
        self.writer.close()
