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

    def log_h_full(self, ICLD, ICTD, _, epoch):
        if epoch % self.h_log_interval != 0:
            return

        # Histograms
        self.writer.add_histogram("ICLD", ICLD.detach(), epoch)
        self.writer.add_histogram("ICTD", ICTD.detach(), epoch)

        ICLD_np = ICLD.detach().cpu().numpy()
        ICTD_np = ICTD.detach().cpu().numpy()

        colors = np.arange(len(ICTD[0,0]))

        fig, ax = plt.subplots()

        sc = ax.scatter(
            ICTD_np[0,0],
            ICLD_np[0,0],
            c=colors,
            cmap='viridis',
            label='Left loudspeaker filter'
        )

        fig.colorbar(sc, ax=ax, label='1/3 octave band index')
        ax.set_title('ICLD/ICTD pairs after training')
        ax.set_xlabel('ICTD [ms]')
        ax.set_ylabel('ICLD [dB]')
        ax.set_ylim(-20, 20)
        ax.set_xlim(-1, 1)
        ax.grid(True)

        # now this works
        self.writer.add_figure("ICLD_ICTD", fig, global_step=epoch)
        plt.close(fig)

    def close(self):
        self.writer.close()
