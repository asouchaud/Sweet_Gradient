import scipy 
import numpy as np
import os 
import argparse

def save_wav(est_filters, fs, wav_path):
    # Normalize filters to prevent clipping
    est_filters /= np.max(np.abs(est_filters)) / 0.9

    # Save as WAV file (stereo)
    scipy.io.wavfile.write(wav_path, fs, est_filters.T.astype(np.float32))


def main(args):
    est_filters = np.load(args.est_filters_path)
    parent_folder_path = os.path.dirname(os.path.dirname(args.est_filters_path))

    fs = args.fs
    wav_file_dir = os.path.join(parent_folder_path, "wav_files")
    os.makedirs(wav_file_dir, exist_ok=True)
    wave_name = os.path.basename(args.est_filters_path).replace(".npy", ".wav")

    wav_file_path = os.path.join(wav_file_dir, wave_name)

    save_wav(est_filters, fs, wav_file_path)


parser = argparse.ArgumentParser(description="Save estimated filters as WAV file")
parser.add_argument("--est_filters_path", type=str, required=True, help="Path to the estimated filters .npy file")
parser.add_argument("--fs", type=int, default=48000, help="Sampling frequency")
args = parser.parse_args()


if __name__ == "__main__":
    main(args)