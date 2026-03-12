from pathlib import Path
import json
import numpy as np
from datetime import datetime
import scipy.io as sio
import matplotlib.pyplot as plt

from scripts.saving.save_normalise import normalise_max


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


    def save_results(self, args, estimation_dict, est_filters, fs, lstn_pos, spk_pos, trainer):    
        # =========================
        # Save results (clean layout)
        # Subfolders

        est_filters = normalise_max(est_filters)

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
            "alpha": trainer.alpha
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
        