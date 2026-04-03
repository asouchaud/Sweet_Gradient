from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from io import BytesIO
import json
import sys
from types import SimpleNamespace
import numpy as np
from datetime import datetime
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

from scripts.plotting.graph_plot import plot_frequency_response, plot_stem_signal, rfft_mag_db_phase, update_stem_artists
from scripts.plotting.results_plot import plot_pair_directivity_polar


def _to_json_safe(obj):
    """Recursively convert numpy/torch/path values so ``json.dump`` can persist them."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return [_to_json_safe(x) for x in obj]
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.numel() <= 4096 else str(obj)
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, torch.dtype):
            return str(obj)
    except ImportError:
        pass
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            arr = np.asarray(obj)
            if arr.ndim == 0:
                return _to_json_safe(arr.item())
        except Exception:
            pass
    return str(obj)


def _coerce_env_to_dict(env):
    """Normalise ``env`` (mapping, dataclass instance, or ``types.SimpleNamespace``) to a plain dict."""
    if env is None:
        return None
    if is_dataclass(env) and not isinstance(env, type):
        return asdict(env)
    if isinstance(env, Mapping):
        return dict(env)
    if isinstance(env, SimpleNamespace):
        return dict(vars(env))
    raise TypeError(
        "env must be a mapping, a dataclass instance, or types.SimpleNamespace; "
        f"got {type(env)!r}"
    )


class saver():
    def __init__(self, loss_dict, lstn_pos):
        # Root experiment directory
        exp_root = Path("results")
        exp_root.mkdir(exist_ok=True)

        # -------------------------
        # Generate run folder name
        # -------------------------
        now = datetime.now()
        date_prefix = now.strftime("%y%m%d")
        time_suffix = now.strftime("%H%M%S")
        entrypoint_stem = Path(sys.argv[0]).stem

        # Build run folder name
        run_name = f"{entrypoint_stem}_{date_prefix}_{time_suffix}"
        run_dir = exp_root / run_name
        
        self.run_dir = run_dir
        self.run_name = run_name
        self.now = now
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def get_plot_folder(self):
        plot_folder = self.run_dir / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
        return plot_folder

    def save_filter_video(
        self,
        h_history,
        source_index: int,
        fs: float,
        fps: int = 20,
        hold_last_seconds: float = 2.0,
        resolution_scale: float = 0.7,
        time_title = None,
        mag_title = "Magnitude",
        phase_title = "Phase",
        max_min = None,
        mag_max = None,
        include_phase_response: bool = True,
    ):
        """
        Save a simple video (GIF) of a single source FIR over epochs.

        Parameters
        ----------
        h_history : list/sequence
            Per-epoch filters, each element shaped (n_sources, n_samples) or (n_samples, n_sources).
        source_index : int
            Index of the source to render.
        fs : float
            Sampling rate in Hz.
        fps : int
            Frames per second for the GIF.
        hold_last_seconds : float
            Extra time (in seconds) to hold the last frame before looping.
        resolution_scale : float
            Scale factor for figure size (and thus output resolution). Use < 1 to reduce
            file size (e.g. 0.5 gives roughly one quarter the pixels).
        time_title : str, optional
            Title prefix for the time-domain panel.
        mag_title : str
            Title for the magnitude-response panel.
        phase_title : str
            Title for the phase-response panel.
        max_min : float or tuple, optional
            Time-domain y-axis limit. A scalar applies symmetric limits (-value, value);
            a tuple applies explicit (min, max) limits.
        mag_max : float or tuple, optional
            Magnitude y-axis limit. A scalar sets the upper limit while keeping the
            computed lower limit; a tuple applies explicit (min, max) limits.
        include_phase_response : bool
            Whether to include the phase-response panel.
        """
        if h_history is None or len(h_history) == 0:
            return

        arr = np.asarray(h_history)
        if arr.ndim != 3:
            raise ValueError(f"h_history must be 3D, got shape {arr.shape}")

        # Make per-epoch mono FIR array: (n_epoch, n_samples)
        if arr.shape[1] <= arr.shape[2]:
            if not 0 <= source_index < arr.shape[1]:
                raise IndexError(f"source_index {source_index} is out of bounds for {arr.shape[1]} sources")
            h_epoch = arr[:, source_index, :]
        else:
            if not 0 <= source_index < arr.shape[2]:
                raise IndexError(f"source_index {source_index} is out of bounds for {arr.shape[2]} sources")
            h_epoch = arr[:, :, source_index]

        max_abs = float(np.max(np.abs(h_epoch))) if np.size(h_epoch) else 1.0
        if max_abs == 0.0:
            max_abs = 1.0
        if max_min is None:
            max_min = 1.05 * max_abs

        video_dir = self.run_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        out_path = video_dir / f"source_{source_index}_filter.gif"

        # Precompute frequency responses for stable axes during animation.
        # Reuse the same helper as `plot_freq(...)` so the setup stays identical.
        n_taps = int(h_epoch.shape[-1])
        n_fft = 4096
        f_hz, mag_db, phase = rfft_mag_db_phase(h_epoch, fs, n_fft=n_fft)

        f_mask = f_hz > 0.0  # exclude DC for log-x plots
        mag_db_plot = mag_db[:, f_mask]
        phase_plot = phase[:, f_mask] if include_phase_response else None

        mag_min = float(np.min(mag_db_plot))
        mag_max_data = float(np.max(mag_db_plot))
        pad = 0.05 * (mag_max_data - mag_min + 1e-9)
        if mag_max is None:
            mag_max = (mag_min - pad, mag_max_data + pad)

        scale = max(0.1, float(resolution_scale))
        if include_phase_response:
            fig, (ax_t, ax_m, ax_p) = plt.subplots(
                nrows=1,
                ncols=3,
                figsize=(12 * scale, 3 * scale),
                gridspec_kw={"width_ratios": [2.0, 1.3, 1.3]},
            )
        else:
            fig, (ax_t, ax_m) = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(8.5 * scale, 3 * scale),
                gridspec_kw={"width_ratios": [2.0, 1.3]},
            )
            ax_p = None

        time_title = f"Source {source_index} FIR" if time_title is None else time_title

        # Time domain (stem plot)
        stem_bottom = 0.0
        _, ax_t, stem_t, x = plot_stem_signal(
            h_epoch[0],
            ax=ax_t,
            title=f"{time_title} - epoch 0",
            max_min=max_min,
            bottom=stem_bottom,
            return_artists=True,
        )

        # Magnitude / phase plots
        _, ax_m, ax_p, line_m, line_p, _f_plot, _phase_plot = plot_frequency_response(
            freqs_hz=f_hz,
            mag_db=mag_db[0],
            phase=phase[0] if include_phase_response else None,
            ax_mag=ax_m,
            ax_phase=ax_p,
            mag_title=mag_title,
            phase_title=phase_title,
            mag_max=mag_max,
            include_phase=include_phase_response,
            return_lines=True,
        )

        fig.tight_layout(rect=[0, 0.08, 1, 1])

        def _update(frame_idx):
            y = h_epoch[frame_idx]
            update_stem_artists(stem_t, x, y, bottom=stem_bottom)
            ax_t.set_title(f"{time_title} - epoch {frame_idx}")
            line_m.set_ydata(mag_db_plot[frame_idx])

            artists = [stem_t.markerline, stem_t.stemlines, line_m]
            if include_phase_response and line_p is not None:
                line_p.set_ydata(phase_plot[frame_idx])
                artists.append(line_p)
            return tuple(artists)

        from matplotlib.animation import FuncAnimation, PillowWriter

        n_frames = len(h_epoch)
        hold_n = int(round(max(0.0, float(hold_last_seconds)) * max(int(fps), 1)))
        frame_indices = list(range(n_frames)) + ([n_frames - 1] * hold_n if n_frames > 0 else [])

        anim = FuncAnimation(
            fig,
            _update,
            frames=frame_indices,
            interval=1000 / max(int(fps), 1),
            # Stem uses a LineCollection; blitting it with Pillow is unreliable.
            blit=False,
        )
        anim.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)

    def save_directivity_video(
        self,
        h_history,
        spk_pos,
        fs,
        group_indices,
        radius=None,
        n_angles=720,
        freqs_eval_hz=(250, 500, 1000, 2000, 4000),
        c=343.0,
        overlay_curve=None,
        overlay_curves=None,
        overlay_labels=None,
        fps: int = 20,
        hold_last_seconds: float = 2.0,
        resolution_scale: float = 0.7,
        db_floor=None,
        overlay_avg_directivity: bool = True,
    ):
        if h_history is None or len(h_history) == 0:
            return

        arr = np.asarray(h_history)
        if arr.ndim != 3:
            raise ValueError(f"h_history must be 3D, got shape {arr.shape}")

        if arr.shape[1] <= arr.shape[2]:
            est_filters_epoch = arr
        else:
            est_filters_epoch = np.swapaxes(arr, 1, 2)

        group_indices = np.asarray(group_indices, dtype=int).reshape(-1)
        if group_indices.size < 1:
            raise ValueError("group_indices must contain at least one loudspeaker index.")
        if np.any(group_indices < 0) or np.any(group_indices >= est_filters_epoch.shape[1]):
            raise IndexError("group_indices contain an out-of-bounds loudspeaker index.")

        video_dir = self.run_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        idx_str = "_".join(str(int(i)) for i in group_indices.tolist())
        out_path = video_dir / f"directivity_sources_{idx_str}.gif"

        dpi = max(20, int(round(100 * max(0.1, float(resolution_scale)))))
        frames = []
        for epoch_idx, est_filters in enumerate(est_filters_epoch):
            fig = plot_pair_directivity_polar(
                est_filters=est_filters,
                spk_pos=spk_pos,
                fs=fs,
                group_indices=group_indices,
                radius=radius,
                n_angles=n_angles,
                freqs_eval_hz=freqs_eval_hz,
                c=c,
                overlay_curve=overlay_curve,
                overlay_curves=overlay_curves,
                overlay_labels=overlay_labels,
                show=False,
                db_floor=db_floor,
                overlay_avg_directivity=overlay_avg_directivity,
            )
            fig.axes[0].set_title(f"{fig.axes[0].get_title()} - epoch {epoch_idx}")

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB").copy())
            buf.close()

        if len(frames) == 0:
            return

        hold_n = int(round(max(0.0, float(hold_last_seconds)) * max(int(fps), 1)))
        frames.extend([frames[-1].copy() for _ in range(hold_n)])
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(round(1000 / max(int(fps), 1))),
            loop=0,
        )


    def save_results(self, estimation_dict, est_filters, fs, lstn_pos, spk_pos, trainer, env=None):
        # =========================
        # Save results (clean layout)
        # Subfolders

        artifacts_dir = self.run_dir / "artifacts"

        artifacts_dir.mkdir(parents=True)

        wav_dir = self.run_dir / "wave_files"

        wav_dir.mkdir(parents=True)

        # ---------- Save artifacts ----------
        np.save(artifacts_dir / "est_filters.npy",
                est_filters)
        
        sio.wavfile.write(wav_dir / "est_filters.wav", fs, est_filters.T)

        # Optional: save perceptual outputs as arrays if needed
        for key, value in estimation_dict.items():
            np.save(artifacts_dir / f"{key}.npy",
                    np.asarray(value))

        # ---------- Config (static experiment definition) ----------
        config = {
            "run_name": self.run_name,
            "date": self.now.strftime("%Y-%m-%d %H:%M:%S"),
            "entrypoint_file": Path(sys.argv[0]).name,
        }
        env_dict = _coerce_env_to_dict(env)
        if env_dict is not None:
            config["main"] = _to_json_safe(env_dict)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # ---------- Results (small, queryable numbers only) ----------
        results = {}
        for key, value in estimation_dict.items():
            try:
                results[key] = np.asarray(value).tolist()
            except Exception:
                results[key] = str(value)
        
        # Save Stimulus
        stim_np = trainer.prop.stim.detach().cpu().numpy().squeeze()
        np.save(artifacts_dir / "stimulus.npy", stim_np)

        # Include final estimated source filters as well (same as artifacts/est_filters.npy)
        try:
            results["est_filters"] = np.asarray(est_filters).tolist()
        except Exception:
            results["est_filters"] = str(est_filters)

        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"\n✅ Results saved to {self.run_dir}")

    def save_plots(self, plots_or_time_plot, freq_plot=None, arrows_plot=None, color_l=None, color_r=None):
        if isinstance(plots_or_time_plot, dict):
            time_plot = plots_or_time_plot.get("time_plot")
            freq_plot = plots_or_time_plot.get("freq_plot")
            stimulus_plot = plots_or_time_plot.get("stimulus_plot")
            arrows_plot = plots_or_time_plot.get("arrow_plot")
            color_l = plots_or_time_plot.get("color_l_plot")
            color_r = plots_or_time_plot.get("color_r_plot")
            directivity_plot = plots_or_time_plot.get("directivity_plot")
            directivity_arrow_plot = plots_or_time_plot.get("arrows_and_directivity")
            directivity_arrow_centre_plot = plots_or_time_plot.get("arrows_and_directivity_centre")
        else:
            time_plot = plots_or_time_plot
            stimulus_plot = None
            directivity_plot = None
            directivity_arrow_plot = None
            directivity_arrow_centre_plot = None

        plot_folder = self.run_dir / "plots"
        plot_folder.mkdir(parents=True)

        if time_plot is not None:
            time_plot.savefig(plot_folder / "time_filters.pdf")
        if freq_plot is not None:
            freq_plot.savefig(plot_folder / "freq_filters.pdf")
        if stimulus_plot is not None:
            stimulus_plot.savefig(plot_folder / "stimulus.pdf")

        if arrows_plot is not None:
            arrows_plot.savefig(plot_folder / "arrow_plot.pdf")

        if color_l is not None and color_r is not None:
            color_l.savefig(plot_folder / "color_left.pdf")
            color_r.savefig(plot_folder / "color_right.pdf")

        if directivity_plot is not None:
            directivity_plot.savefig(plot_folder / "directivity_plot.pdf")
        if directivity_arrow_plot is not None:
            directivity_arrow_plot.savefig(
                plot_folder / "arrows_and_directivity.pdf",
                bbox_inches="tight",
                pad_inches=0.06,
            )
        if directivity_arrow_centre_plot is not None:
            directivity_arrow_centre_plot.savefig(
                plot_folder / "arrows_and_directivity_centre.pdf",
                bbox_inches="tight",
                pad_inches=0.06,
            )