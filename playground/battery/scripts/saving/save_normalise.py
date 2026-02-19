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

    tmp = np.load(os.path.join(exp_path, pos_list[0], angle_list[0], "np_files", exp_list[0]))

    new_filters = np.zeros((len(pos_list), len(angle_list), len(exp_list), tmp.shape[0], tmp.shape[1])) # Assuming 2 channels and 256 taps, adjust if different

    for pos_id, pos_str in enumerate(pos_list):
        for angle_id, angle_str in enumerate(angle_list):
            for exp_id, exp_str in enumerate(exp_list):
                filt_path = os.path.join(exp_path, pos_str, angle_str, "np_files", exp_str)
                filts = np.load(filt_path)

                filts_rms = np.sqrt(np.mean(filts**2))
                filts_normed = filts/(filts_rms + 1e-8)
                new_filters[pos_id, angle_id, exp_id] = filts_normed

    new_filters = new_filters * 0.9 / np.max(np.abs(new_filters))     

    for pos_id, pos_str in enumerate(pos_list):
        for angle_id, angle_str in enumerate(angle_list):
            for exp_id, exp_str in enumerate(exp_list):
                filt_path = os.path.join(exp_path, pos_str, angle_str, "np_files", exp_str)
                filt = new_filters[pos_id, angle_id, exp_id]

                np.save(filt_path, filt)

                sio.wavfile.write(os.path.join(exp_path, pos_str, angle_str, "wav_files", exp_str.replace(".npy", ".wav")), 48000, filt.T.astype(np.float32))


    
argparser = argparse.ArgumentParser()
argparser.add_argument("--path", type=str, default= "playground/battery/results/r0_i2_n64_260216_102400",help="Path to the experiment folder containing the position and angle subfolders with the np_files and wav_files folders inside.")
args = argparser.parse_args()

if __name__ == "__main__":
    save_symetric(args)