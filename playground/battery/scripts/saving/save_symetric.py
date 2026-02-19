import numpy as np
import scipy.io as sio
import pyloudnorm as ply
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import argparse

def save_symetric(args):

    exp_path = args.path 

    # Extract positions, angles and optimisations from the experiment path
    pos_list = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
    angle_list = [name for name in os.listdir(os.path.join(exp_path, pos_list[0])) if os.path.isdir(os.path.join(exp_path, pos_list[0], name))] 
    exp_list = [name for name in os.listdir(os.path.join(exp_path, pos_list[0], angle_list[0], "np_files")) if os.path.isfile(os.path.join(exp_path, pos_list[0], angle_list[0], "np_files", name))]

    # Convert angle folder names to numerical values (assuming they are in the format "angle_XX")
    angle_values = [int(s.split('_')[0]) % 360 for s in angle_list]

    # Find all non-zero symmetric pairs
    symmetric_pairs = {a for a in angle_values if a >= 0 and (360 - a) in angle_values or a == 0}
    symmetric_angles = symmetric_pairs.union({(-a) % 360 for a in symmetric_pairs})

    # Build list without symmetric angles
    angles_without_symmetric = [a for a in angle_values if a not in symmetric_angles]

    # Create their symmetric counterparts (mod 360)
    symmetric_of_remaining = [( -a ) % 360 for a in angles_without_symmetric]

    # Create a mapping of angles to their symmetric counterparts
    old_angles_str = [f"{a % 360:03d}_deg" for a in angles_without_symmetric]
    new_angles_str = [f"{a % 360:03d}_deg" for a in symmetric_of_remaining]

    for pos_str in pos_list:
        for i, angle_str in enumerate(new_angles_str):
            os.mkdir(os.path.join(exp_path, pos_str, angle_str))
            os.mkdir(os.path.join(exp_path, pos_str, angle_str, "np_files"))
            os.mkdir(os.path.join(exp_path, pos_str, angle_str, "wav_files"))

            for exp_str in exp_list:
                filt_path = os.path.join(exp_path, pos_str, old_angles_str[i], "np_files", exp_str)
                original_filters = np.load(filt_path)

                symmetric_filt = np.flip(original_filters, axis=0) # Flip the filters channels to create the symmetric counterpart

                # Save the symmetric filters in the new angle folder
                new_filt_path = os.path.join(exp_path, pos_str, angle_str, "np_files", exp_str)
                np.save(new_filt_path, symmetric_filt)

                sio.wavfile.write(os.path.join(exp_path, pos_str, angle_str, "wav_files", exp_str.replace(".npy", ".wav")), 48000, symmetric_filt.T.astype(np.float32))


    
argparser = argparse.ArgumentParser()
argparser.add_argument("--path", type=str, default= "playground/battery/results/r0_i2_n64_260216_102400",help="Path to the experiment folder containing the position and angle subfolders with the np_files and wav_files folders inside.")
args = argparser.parse_args()

if __name__ == "__main__":
    save_symetric(args)