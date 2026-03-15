from pathlib import Path
import json
import numpy as np
from datetime import datetime
import scipy.io as sio
import matplotlib.pyplot as plt

from scripts.plotting.graph_plot import rfft_mag_db_phase, plot_mag_db_from_response


class saver():
    def __init__(self, perceptual_dict, lstn_pos, args):
        # Root experiment directory
        exp_root = Path("results")
        exp_root.mkdir(exist_ok=True)

        # -------------------------
        # Generate run folder name
        # -------------------------
        now = datetime.now()
        date_prefix = now.strftime("%y%m%d")
        time_suffix = now.strftime("%H%M%S")

        # Alphabetical flags from perceptual_dict keys
        flag_str = ''.join(sorted(k[0] for k in perceptual_dict.keys()))

        # Number of listener positions
        n_listeners = len(lstn_pos)

        reg_map = {
            0: "0", # No regularisation
            1: "1", # First moment regularisation
            2: "2", # Second moment regularisation
        }

        init_map = {
            0: "n", # Random noise initialisation
            1: "2", # Impulses at both loudspeakers initialisation
            2: "1" # Impulse at only left loudspeaker and silent right loudspeaker initialisation
        }

        field_map = {
            0: "ff", # Far-field
            1: "nf", # Near-field
        }

        # Build run folder name
        run_name = f"{flag_str}_n{n_listeners}_r{reg_map[args.reg]}_i{init_map[args.init]}_{field_map[args.near_field]}_g{args.gradnorm}_{date_prefix}_{time_suffix}"
        run_dir = exp_root / run_name
        
        self.run_dir = run_dir
        self.run_name = run_name
        self.now = now
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_filter_video(
        self,
        h_history,
        source_index: int,
        fs: float,
        fps: int = 20,
        hold_last_seconds: float = 2.0,
        resolution_scale: float = 0.7,
    ):
        """
        Save a simple video (GIF) of a single source FIR over epochs.

        Parameters
        ----------
        h_history : list/sequence
            Per-epoch filters, each element shaped (2, n_samples) or (n_samples, 2).
        source_index : int
            0 for left, 1 for right.
        fs : float
            Sampling rate in Hz.
        fps : int
            Frames per second for the GIF.
        hold_last_seconds : float
            Extra time (in seconds) to hold the last frame before looping.
        resolution_scale : float
            Scale factor for figure size (and thus output resolution). Use < 1 to reduce
            file size (e.g. 0.5 gives roughly one quarter the pixels).
        """
        if h_history is None or len(h_history) == 0:
            return

        arr = np.asarray(h_history)
        if arr.ndim != 3:
            raise ValueError(f"h_history must be 3D, got shape {arr.shape}")

        # Make per-epoch mono FIR array: (n_epoch, n_samples)
        if arr.shape[1] == 2:
            h_epoch = arr[:, source_index, :]
        elif arr.shape[2] == 2:
            h_epoch = arr[:, :, source_index]
        else:
            raise ValueError(f"Expected a stereo filter with a dimension of size 2, got shape {arr.shape}")

        max_abs = float(np.max(np.abs(h_epoch))) if np.size(h_epoch) else 1.0
        if max_abs == 0.0:
            max_abs = 1.0

        video_dir = self.run_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        out_path = video_dir / f"source_{source_index}_filter.gif"

        # Precompute frequency responses for stable axes during animation.
        # Reuse the same helper as `plot_freq(...)` so the setup stays identical.
        n_taps = int(h_epoch.shape[-1])
        n_fft = 4096
        f_hz, mag_db, phase = rfft_mag_db_phase(h_epoch, fs, n_fft=n_fft)

        f_mask = f_hz > 0.0  # exclude DC for log-x plots
        f_plot = f_hz[f_mask]
        mag_db_plot = mag_db[:, f_mask]
        phase_plot = phase[:, f_mask]

        scale = max(0.1, float(resolution_scale))
        fig, (ax_t, ax_m, ax_p) = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(12 * scale, 3 * scale),
            gridspec_kw={"width_ratios": [2.0, 1.3, 1.3]},
        )

        # Time domain
        x = np.arange(n_taps)
        (line_t,) = ax_t.plot(x, h_epoch[0], lw=1.5)
        ax_t.set_xlim(0, len(x) - 1)
        ax_t.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
        ax_t.set_xlabel("Sample")
        ax_t.set_ylabel("Amplitude")
        ax_t.set_title(f"Source {source_index} FIR — epoch 0")
        ax_t.grid(True, alpha=0.25)

        # Magnitude (reuse same plotting helper as `plot_freq`)
        _, _, line_m, _ = plot_mag_db_from_response(f_hz, mag_db[0], ax=ax_m, title="Magnitude", return_line=True)
        line_m.set_ydata(mag_db_plot[0])
        mag_min = float(np.min(mag_db_plot))
        mag_max = float(np.max(mag_db_plot))
        pad = 0.05 * (mag_max - mag_min + 1e-9)
        ax_m.set_ylim(mag_min - pad, mag_max + pad)

        # Phase
        (line_p,) = ax_p.plot(f_plot, phase_plot[0], lw=1.2)
        ax_p.set_xscale("log")
        ax_p.set_xlim(float(f_plot[0]), float(f_plot[-1]))
        ax_p.set_ylim(-np.pi, np.pi)
        ax_p.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
        ax_p.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax_p.set_xlabel("Frequency (Hz)")
        ax_p.set_ylabel("Phase (rad)")
        ax_p.set_title("Phase")
        ax_p.grid(True, alpha=0.25)

        fig.tight_layout(rect=[0, 0.08, 1, 1])

        def _update(frame_idx):
            line_t.set_ydata(h_epoch[frame_idx])
            ax_t.set_title(f"Source {source_index} FIR — epoch {frame_idx}")
            line_m.set_ydata(mag_db_plot[frame_idx])
            line_p.set_ydata(phase_plot[frame_idx])
            return (line_t, line_m, line_p)

        from matplotlib.animation import FuncAnimation, PillowWriter

        n_frames = len(h_epoch)
        hold_n = int(round(max(0.0, float(hold_last_seconds)) * max(int(fps), 1)))
        frame_indices = list(range(n_frames)) + ([n_frames - 1] * hold_n if n_frames > 0 else [])

        anim = FuncAnimation(
            fig,
            _update,
            frames=frame_indices,
            interval=1000 / max(int(fps), 1),
            blit=True,
        )
        anim.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)


    def save_results(self, args, estimation_dict, est_filters, fs, lstn_pos, spk_pos, trainer):    
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
            "n_epochs": args.epochs,
            "seed": args.seed,
            "fs": fs,
            "listeners_position": lstn_pos,
            "n_listeners": len(lstn_pos),
            "speakers_position": spk_pos,
            "near_field": bool(args.near_field),
            "target_angle": args.target_angle,
            "stimulus": args.stimulus,
            "HRTF": Path(args.sofa_path).name,
            "learning_rate": args.lr,
            "init": args.init,
            "regularisation": args.reg,
            "perceptual_losses": list(estimation_dict.keys()),
            "Gradnorm": args.gradnorm,
            "importance": trainer.importance,
            "alpha": trainer.alpha,
            "weights": {k: v.detach().cpu().item() for k, v in trainer.loss_weights.items()}
        }

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # ---------- Results (small, queryable numbers only) ----------
        results = {}
        for key, value in estimation_dict.items():
            try:
                results[key] = np.asarray(value).tolist()
            except Exception:
                results[key] = str(value)

        # Include final estimated source filters as well (same as artifacts/est_filters.npy)
        try:
            results["est_filters"] = np.asarray(est_filters).tolist()
        except Exception:
            results["est_filters"] = str(est_filters)

        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"\n✅ Results saved to {self.run_dir}")

    def save_plots(self, time_plot, freq_plot, arrows_plot = None, color_l = None, color_r = None):

        plot_folder = self.run_dir / "plots"
        plot_folder.mkdir(parents=True)

        time_plot.savefig(plot_folder / "time_filters.png")
        freq_plot.savefig(plot_folder / "freq_filters.png")

        if arrows_plot is not None:
            arrows_plot.savefig(plot_folder / "arrow_plot.png")

        if color_l is not None and color_r is not None:
            color_l.savefig(plot_folder / "color_left.png")
            color_r.savefig(plot_folder / "color_right.png")
        