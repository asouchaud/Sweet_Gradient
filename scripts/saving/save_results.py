from pathlib import Path
import json
import numpy as np
from datetime import datetime


def save_results(args, estimation_dict, est_filters, fs, lstn_pos, spk_pos):    
    # =========================
    # Save results (clean layout)
    # =========================

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
    flag_str = ''.join(sorted(k[0] for k in estimation_dict.keys()))

    # Number of listener positions
    n_listeners = len(lstn_pos)

    # Build run folder name
    run_name = f"{date_prefix}_{flag_str}_n{n_listeners}_{time_suffix}"
    run_dir = exp_root / run_name

    # Subfolders
    artifacts_dir = run_dir / "artifacts"

    artifacts_dir.mkdir(parents=True)

    # ---------- Save artifacts ----------
    np.save(artifacts_dir / "est_filters.npy",
            est_filters)

    # Optional: save perceptual outputs as arrays if needed
    for key, value in estimation_dict.items():
        np.save(artifacts_dir / f"{key}.npy",
                np.asarray(value))

    # ---------- Config (static experiment definition) ----------
    config = {
        "run_name": run_name,
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "n_epochs": args.epochs,
        "seed": args.seed,
        "fs": fs,
        "listeners_position": lstn_pos,
        "n_listeners": n_listeners,
        "speakers_position": spk_pos,
        "target_angle": args.target_angle,
        "stimulus": args.stimulus,
        "HRTF": Path(args.sofa_path).name,
        "learning_rate": args.lr,
        "init": args.init,
        "regularisation": args.reg,
        "perceptual_losses": list(estimation_dict.keys())
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # ---------- Results (small, queryable numbers only) ----------
    results = {}
    for key, value in estimation_dict.items():
        try:
            results[key] = np.asarray(value).tolist()
        except Exception:
            results[key] = str(value)

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Results saved to {run_dir}")
