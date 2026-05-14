import numpy as np
import matplotlib.pyplot as plt


def plot_localisation_gradient_norms(localisation_grad_tracking):
    if localisation_grad_tracking is None:
        return None

    epochs = np.asarray(localisation_grad_tracking["epoch"])
    itd_grad_norm = np.asarray(localisation_grad_tracking["itd_grad_norm"])
    ild_grad_norm = np.asarray(localisation_grad_tracking["ild_grad_norm"])
    if epochs.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, itd_grad_norm, label="||dL/dITD||", linewidth=2)
    ax.plot(epochs, ild_grad_norm, label="||dL/dILD||", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm (L2)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_localisation_gradient_band_contributions(localisation_grad_tracking):
    if localisation_grad_tracking is None:
        return None

    epochs = np.asarray(localisation_grad_tracking["epoch"])
    itd_grad_band_norm = np.asarray(localisation_grad_tracking["itd_grad_band_norm"], dtype=float)
    ild_grad_band_norm = np.asarray(localisation_grad_tracking["ild_grad_band_norm"], dtype=float)
    itd_grad_band_share = np.asarray(localisation_grad_tracking["itd_grad_band_share"], dtype=float)
    ild_grad_band_share = np.asarray(localisation_grad_tracking["ild_grad_band_share"], dtype=float)
    if epochs.size == 0 or itd_grad_band_norm.size == 0 or ild_grad_band_norm.size == 0:
        return None

    eps = 1e-12
    itd_log = np.log10(np.clip(itd_grad_band_norm, eps, None))
    ild_log = np.log10(np.clip(ild_grad_band_norm, eps, None))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)

    im0 = axes[0, 0].imshow(itd_log.T, origin="lower", aspect="auto")
    axes[0, 0].set_title("ITD band gradient norm (log10)")
    axes[0, 0].set_ylabel("Critical band index")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(ild_log.T, origin="lower", aspect="auto")
    axes[0, 1].set_title("ILD band gradient norm (log10)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(itd_grad_band_share.T, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("ITD band contribution share")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Critical band index")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(ild_grad_band_share.T, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("ILD band contribution share")
    axes[1, 1].set_xlabel("Epoch")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def plot_localisation_cue_dominance(localisation_grad_tracking):
    if localisation_grad_tracking is None:
        return None

    itd_grad_band_norm = np.asarray(localisation_grad_tracking["itd_grad_band_norm"], dtype=float)
    ild_grad_band_norm = np.asarray(localisation_grad_tracking["ild_grad_band_norm"], dtype=float)
    if itd_grad_band_norm.size == 0 or ild_grad_band_norm.size == 0:
        return None

    denom = itd_grad_band_norm + ild_grad_band_norm + 1e-12
    itd_share_joint = itd_grad_band_norm / denom
    ild_share_joint = ild_grad_band_norm / denom
    cue_dominance = (itd_grad_band_norm - ild_grad_band_norm) / denom
    cue_winner = (itd_grad_band_norm >= ild_grad_band_norm).astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)

    im0 = axes[0, 0].imshow(itd_share_joint.T, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("ITD share within each band")
    axes[0, 0].set_ylabel("Critical band index")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(ild_share_joint.T, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("ILD share within each band")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(cue_dominance.T, origin="lower", aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1, 0].set_title("Cue dominance: +ITD / -ILD")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Critical band index")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(cue_winner.T, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="PiYG")
    axes[1, 1].set_title("Winning cue (1=ITD, 0=ILD)")
    axes[1, 1].set_xlabel("Epoch")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig
