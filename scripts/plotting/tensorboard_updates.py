import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from scripts.plotting.graph_plot import plot_time
from scripts.plotting.graph_plot import plot_freq
from scripts.plotting.graph_plot import plot_arrows
from scripts.plotting.graph_plot import plot_bin_cues

class TensorboardLogger:
    def __init__(self, log_dir="runs/train", h_log_interval=50, fs = 48000, device='cuda', dtype=torch.float32):
        from scripts.setup.setup import load_gtfb
        self.fs = fs
        self.writer = SummaryWriter(log_dir=log_dir)
        self.h_log_interval = h_log_interval
        self.gtfb = load_gtfb(fs, device=device, dtype=dtype)


    def log_losses(self, loss_dict, full_loss, epoch):
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"loss/{key}", value.item(), global_step=epoch)
        self.writer.add_scalar("loss/total", full_loss.item(), global_step=epoch)


    def log_h_full(self, h_full, epoch):
        if epoch % self.h_log_interval != 0:
            return
        
        h_full = h_full.squeeze(0).detach().cpu().numpy()

        # Histograms
        for source_idx, source_filter in enumerate(h_full):
            self.writer.add_histogram(f"h_full/source_{source_idx}", source_filter, global_step=epoch)

        # Time domain plots
        fig_time = plot_time(h_full, show=False)
        self.writer.add_figure("h_full (Time Domain)", fig_time, global_step=epoch)
        plt.close(fig_time)

        # Frequency domain plots
        fig_freq = plot_freq(h_full, self.fs, show=False)
        self.writer.add_figure("h_full (Frequency Domain)", fig_freq, global_step=epoch)
        plt.close(fig_freq)
    

    def log_bin_cues(self, bin_stim, epoch):
        if epoch % self.h_log_interval != 0:
            return

        fig_bin_cues = plot_bin_cues(bin_stim, self.fs, gtfb=self.gtfb, show=False)
        self.writer.add_figure("Binaural Cues", fig_bin_cues, global_step=epoch)
        plt.close(fig_bin_cues)


    def log_plot_arrows(self, lstn_pos, spk_pos, loc, unc = None, col=None, epoch=None, target_pos = None):
        if epoch % self.h_log_interval != 0:
            return
        
        fig_arrows = plot_arrows(lstn_pos, spk_pos, loc, unc, col, target_pos, show=False)

        self.writer.add_figure("Arrow Plot", fig_arrows, global_step=epoch)
        plt.close(fig_arrows)

    def log_weights(self, weight_dict, epoch):
        for key, weight in weight_dict.items():
            self.writer.add_scalar(f'loss_weights/{key}', weight.item(), global_step=epoch)

    def log_r(self, r, epoch):
        for key, ratio in r.items():
             self.writer.add_scalar(f'Relative inverse training rate/{key}', ratio.item(), global_step=epoch)



    def close(self):
        self.writer.close()
