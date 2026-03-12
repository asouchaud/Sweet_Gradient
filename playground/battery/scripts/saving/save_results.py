from pathlib import Path
import numpy as np 
import json
from datetime import datetime
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

class saver():
    def __init__(self, current_date, current_time, reg_idx, init_idx, n_samples, nf_idx, angle, lstn_len, curr_idx):

            # ---------- Index mappings ----------

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

        lstn_map = {
            1: "1_pos",
            3: "3_pos",
            5: "5_pos",
            9: "9_pos",
        }

        curr_map = {
            0: ("lxx.npy", "lxx.json", "lxx.wav"),  # Localisation 
            1: ("xux.npy", "xux.json", "xux.wav"),  # Uncertainty 
            2: ("xxc.npy", "xxc.json", "xxc.wav"),  # Colouration
            3: ("lux.npy", "lux.json", "lux.wav"),  # Localisation and Uncertainty
            4: ("lxc.npy", "lxc.json", "lxc.wav"),  # Localisation and Colouration
            5: ("xuc.npy", "xuc.json", "xuc.wav"),  # Uncertainty and Colouration
            6: ("luc.npy", "luc.json", "luc.wav"),  # Localisation, Uncertainty and Colouration
        }



            # ---------- Resolve folders / filenames ----------
        try:
            reg_id = reg_map[reg_idx]
        except KeyError:
            raise ValueError(f"Invalid reg_idx: {reg_idx}")
        
        try:
            init_id = init_map[init_idx]
        except KeyError:
            raise ValueError(f"Invalid init_idx:{init_idx}")
        
        try:
            field_id = field_map[nf_idx]
        except KeyError:
            raise ValueError(f"Invalid nf_idx: {nf_idx}")

        try:
            pos_folder = lstn_map[lstn_len]
        except KeyError:
            raise ValueError(f"Invalid lstn_len: {lstn_len}")

        try:
            angle_folder = self.angle_to_str(angle)
        except KeyError:
            raise ValueError(f"Invalid angle_idx: {angle}")

        try:
            file_name, json_name, wav_name = curr_map[curr_idx]
        except KeyError:
            raise ValueError(f"Invalid curr_idx: {curr_idx}")

        folder_name = f"r{reg_id}_i{init_id}_n{n_samples}_{field_id}_{current_date}_{current_time}"
        
        # ---------- Base path ----------
        # ---------- Resolve base directory (battery/) ----------
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        folder_path = BASE_DIR / "results" / folder_name
        run_dir = folder_path / pos_folder / angle_folder
        run_dir.mkdir(parents=True, exist_ok=True)

        self.folder_path = folder_path
        self.run_dir = run_dir

        self.file_name = file_name
        self.json_name = json_name
        self.wav_name = wav_name

        (self.run_dir / "wav_files").mkdir(exist_ok=True)
        (self.run_dir / "np_files").mkdir(exist_ok=True)
        (self.run_dir / "json_files").mkdir(exist_ok=True)

    def angle_to_str(self, angle):
        angle_360 = angle % 360
        return f"{int(angle_360):03d}_deg"

    def save_results(self, estimation_dict, est_filters):    

        # ---------- Save wav file ---------------
        maxval = np.max(np.abs(est_filters))
        if maxval > 0:
            est_filters = est_filters / maxval

        wav_path = self.run_dir / "wav_files" / self.wav_name
        wav.write(wav_path, 48000, (est_filters.T * 32767).astype(np.int16))

        # ---------- Save artifacts ----------
        np.save(self.run_dir / "np_files" / self.file_name, est_filters)

        # ---------- Results (small, queryable numbers only) ----------
        results = {}
        for key, value in estimation_dict.items():
            try:
                results[key] = np.asarray(value).tolist()
            except Exception as e:
                results[key] = str(value)

        with open(self.run_dir / "json_files" / self.json_name, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\n✅ Results saved to {self.run_dir / 'json_files' / self.json_name}")
    
    def save_config(self, nf, epochs, n_samples, stimulus, sofa_path, lr, init, reg, seed, log):
        
        config = {
            "near_field": nf,
            "epochs": epochs,
            "n_samples": n_samples,
            "stimulus": stimulus,
            "sofa_path": sofa_path,
            "learning_rate": lr,
            "initialisation": init,
            "regularisation": reg,
            "seed": seed,
            "log": log
        }

        with open(self.folder_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        print(f"\n✅ Config saved to {self.folder_path / 'config.json'}")

    
    def save_plots(self, time_plot, freq_plot, arrows_plot = None, color_l = None, color_r = None):

        plot_folder = self.run_dir / "plots" / self.file_name[:3]
        plot_folder.mkdir(parents=True, exist_ok=True)

        time_plot.savefig(plot_folder / "time_filters.png")
        plt.close(time_plot)  # Close after saving

        freq_plot.savefig(plot_folder / "freq_filters.png")
        plt.close(freq_plot)

        if arrows_plot is not None:
            arrows_plot.savefig(plot_folder / "arrow_plot.png")
            plt.close(arrows_plot)

        if color_l is not None and color_r is not None:
            color_l.savefig(plot_folder / "color_left.png")
            plt.close(color_l)

            color_r.savefig(plot_folder / "color_right.png")
            plt.close(color_r)
        
    

