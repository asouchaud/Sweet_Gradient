from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def save_combined_distance_evolution_animation(localisation_tracking, output_path, fps=10):
    if localisation_tracking is None:
        return None

    distance_combined = np.asarray(localisation_tracking.get("distance_combined"))
    template_azimuths = np.asarray(localisation_tracking.get("template_azimuths"))
    target_azimuths = np.asarray(localisation_tracking.get("target_azimuths"))

    if distance_combined.size == 0:
        return None
    if distance_combined.ndim != 4:
        raise ValueError(
            "Expected distance_combined with shape (n_epochs, n_listeners, n_azimuths, n_bands)."
        )

    n_epochs, n_listeners, n_azimuths, _n_bands = distance_combined.shape
    if template_azimuths.ndim != 1 or template_azimuths.shape[0] != n_azimuths:
        raise ValueError("template_azimuths must be 1D and match distance_combined azimuth dimension.")
    if target_azimuths.size == 0:
        target_azimuths = None
    else:
        target_azimuths = np.squeeze(target_azimuths)
        if target_azimuths.ndim == 0:
            target_azimuths = np.full((n_listeners,), float(target_azimuths), dtype=np.float64)
        elif target_azimuths.ndim != 1 or target_azimuths.shape[0] != n_listeners:
            target_azimuths = None

    cols = int(np.ceil(np.sqrt(n_listeners)))
    rows = int(np.ceil(n_listeners / cols))

    vmin = float(np.min(distance_combined))
    vmax = float(np.max(distance_combined))

    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()

    images = []
    target_lines = []
    for listener_idx in range(n_listeners):
        ax = axes_flat[listener_idx]
        im = ax.imshow(
            distance_combined[0, listener_idx].T,
            aspect="auto",
            origin="lower",
            extent=[template_azimuths[0], template_azimuths[-1], 0, distance_combined.shape[-1] - 1],
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Listener {listener_idx + 1}")
        ax.set_xlabel("Template azimuth (deg)")
        ax.set_ylabel("Critical band index")
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(axis="both", labelsize=8)
        images.append(im)
        if target_azimuths is not None:
            target_line = ax.axvline(
                target_azimuths[listener_idx],
                color="cyan",
                linestyle="--",
                linewidth=1.4,
                alpha=0.95,
            )
            target_lines.append(target_line)

    for i in range(n_listeners, len(axes_flat)):
        axes_flat[i].set_visible(False)

    cbar = fig.colorbar(images[0], ax=axes_flat[:n_listeners], shrink=0.90, pad=0.03, fraction=0.045)
    cbar.set_label("distance['combined']")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("distance['combined']", size=9)

    def _update(frame_idx):
        artists = []
        for listener_idx in range(n_listeners):
            images[listener_idx].set_data(distance_combined[frame_idx, listener_idx].T)
            artists.append(images[listener_idx])
        fig.suptitle(f"distance['combined'] evolution (epoch {frame_idx + 1}/{n_epochs})", fontsize=11)
        return tuple(artists)

    fig.suptitle("distance['combined'] evolution", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.08, top=0.90, wspace=0.34, hspace=0.40)

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_epochs,
        interval=1000 / max(int(fps), 1),
        blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path
