from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def save_itd_ild_evolution_animation(localisation_tracking, output_path, fps=10):
    if localisation_tracking is None:
        return None

    itd = np.asarray(localisation_tracking.get("itd"))
    ild = np.asarray(localisation_tracking.get("ild"))
    target_itd = np.asarray(localisation_tracking.get("target_itd"))
    target_ild = np.asarray(localisation_tracking.get("target_ild"))

    if itd.size == 0 or ild.size == 0:
        return None

    if itd.ndim != 3 or ild.ndim != 3:
        raise ValueError("Expected per-epoch ITD/ILD arrays with shape (n_epochs, n_listeners, n_bands).")
    if target_itd.ndim != 2 or target_ild.ndim != 2:
        raise ValueError("Expected target ITD/ILD arrays with shape (n_listeners, n_bands).")

    n_epochs, n_listeners, n_bands = itd.shape

    itd_ms = itd * 1e3
    target_itd_ms = target_itd * 1e3

    fig, ax = plt.subplots(figsize=(7, 6))
    band_idx = np.arange(n_bands)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=max(n_bands - 1, 1))
    band_colors = cmap(norm(band_idx))

    trail_lines = []
    current_points = []
    target_points = []
    trail_length = 10
    trail_alpha_min = 0.10
    trail_alpha_max = 0.45
    for listener_idx in range(n_listeners):
        listener_trail_lines = []
        listener_current_points = []
        listener_target_points = []
        for band in range(n_bands):
            color = band_colors[band]
            band_trail_lines = []
            for trail_step in range(trail_length):
                alpha = trail_alpha_max - (trail_alpha_max - trail_alpha_min) * (trail_step / max(trail_length - 1, 1))
                (trail_line,) = ax.plot([], [], color=color, linewidth=1.0, alpha=alpha)
                band_trail_lines.append(trail_line)
            (point,) = ax.plot([], [], marker="o", color=color, markersize=4, linestyle="None")
            (target,) = ax.plot(
                [target_itd_ms[listener_idx, band]],
                [target_ild[listener_idx, band]],
                marker="x",
                color=color,
                markersize=6,
                linestyle="None",
            )
            listener_trail_lines.append(band_trail_lines)
            listener_current_points.append(point)
            listener_target_points.append(target)
        trail_lines.append(listener_trail_lines)
        current_points.append(listener_current_points)
        target_points.append(listener_target_points)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-20.0, 20.0)
    ax.set_xlabel("ITD (ms) (+ left leads)")
    ax.set_ylabel("ILD (dB) (+ left louder)")
    ax.set_title("ITD/ILD evolution over epochs")
    ax.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Gammatone filter bank index")

    def _update(frame_idx):
        artists = []
        for listener_idx in range(n_listeners):
            for band in range(n_bands):
                x_hist = itd_ms[: frame_idx + 1, listener_idx, band]
                y_hist = ild[: frame_idx + 1, listener_idx, band]
                for trail_step, trail_line in enumerate(trail_lines[listener_idx][band]):
                    hist_idx = frame_idx - trail_step
                    if hist_idx > 0:
                        trail_line.set_data(
                            [itd_ms[hist_idx - 1, listener_idx, band], itd_ms[hist_idx, listener_idx, band]],
                            [ild[hist_idx - 1, listener_idx, band], ild[hist_idx, listener_idx, band]],
                        )
                    else:
                        trail_line.set_data([], [])
                    artists.append(trail_line)
                current_points[listener_idx][band].set_data([x_hist[-1]], [y_hist[-1]])
                artists.extend((current_points[listener_idx][band], target_points[listener_idx][band]))
        ax.set_title(f"ITD/ILD evolution over epochs (epoch {frame_idx + 1}/{n_epochs})")
        return tuple(artists)

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
