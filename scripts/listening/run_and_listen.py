import argparse
import time
import wave
import pygame
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import sounddevice as sd
from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.datasets.listener_dataset import get_listener_loader
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.setup.setup import set_seed, build_eval_objects
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.perceptual_models.ipd.ipd import interaural_phase_difference_matching
from scripts.optimisation.regularisation import second_moment_regularisation
from scripts.train.train import ListenerBasedLoss, Train

# filepath: scripts/listening/run_and_listen.py

import torch.nn.functional as F

try:
    HAS_SOUNDEVICE = True
except ImportError:
    HAS_SOUNDEVICE = False






def parse_args():
    parser = argparse.ArgumentParser(
        description="Run optimisation and render estimated loudspeaker signals for listening."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_filter_taps", type=int, default=64)
    parser.add_argument("--stimulus", type=str, default="impulse")
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--target_angle_deg", type=float, default=15.0)
    parser.add_argument(
        "--output",
        type=str,
        default="estimated_loudspeaker_output.wav",
        help="WAV file path for the rendered loudspeaker output.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the rendered loudspeaker output after optimisation.",
    )
    return parser.parse_args()


def causal_conv1d(input, weight):
    K = weight.shape[-1]
    pad_input = np.pad(input, (K-1, 0))
    out = np.convolve(pad_input, weight, mode="valid") 
    return out


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_env(args):
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = SimpleNamespace(
        epochs=args.epochs,
        fs=args.fs,
        n_filter_taps=args.n_filter_taps,
        selected_stim=args.stimulus,
        stimulus_batch_size=1,
        n_stimulus=1,
        listener_batch_size=1,
        symmetric_listener_sampling=False,
        target_pos=[
            100.0,
            float(np.tan(np.deg2rad(args.target_angle_deg)) * 100.0),
        ],
        sofa_path=str(
            repo_root
            / "data"
            / "impulses"
            / "SOFA"
            / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
        ),
        lr=args.lr,
        seed=args.seed,
        log=0,
        device_str="cpu",
    )
    env.dtype = torch.float64
    env.device = torch.device(env.device_str)

    env.lstn_pos = torch.tensor([[0.0, 0.0]], device=env.device, dtype=env.dtype)

    base_angle_rad = np.deg2rad(30.0)
    env.speaker_distance = 2.0
    env.spk_pos = torch.tensor(
        [
            [
                env.speaker_distance * np.cos(base_angle_rad),
                env.speaker_distance * np.sin(base_angle_rad),
            ],
            [
                env.speaker_distance * np.cos(base_angle_rad),
                -env.speaker_distance * np.sin(base_angle_rad),
            ],
        ],
        device=env.device,
        dtype=env.dtype,
    )
    env.n_sources = len(env.spk_pos)

    return env


def initialise_models(env):
    stereo_prop = BinauralFreeFieldPropagation(
        fs=env.fs,
        lstn_pos=env.lstn_pos,
        spk_pos=env.spk_pos,
        sofa_file_path=env.sofa_path,
        device=env.device,
        dtype=env.dtype,
    )

    loc_model = llado2025(
        loss=F.mse_loss,
        target_pos=env.target_pos,
        fs=env.fs,
        sofa_path=env.sofa_path,
        lstn_pos=env.lstn_pos,
        spk_pos=env.spk_pos,
        device=env.device,
        dtype=env.dtype,
    )

    col_model = spectral_matching(
        loss=F.l1_loss,
        target=0,
        target_pos=env.target_pos,
        fs=env.fs,
        sofa_path=env.sofa_path,
        lstn_pos=env.lstn_pos,
        device=env.device,
        dtype=env.dtype,
    )

    unc_model = desena2020(
        loss=F.l1_loss,
        target=0,
        fs=env.fs,
        sofa_path=env.sofa_path,
        lstn_pos=env.lstn_pos,
        device=env.device,
        dtype=env.dtype,
    )

    ipd_model = interaural_phase_difference_matching(
        loss=F.l1_loss,
        target=0,
        target_pos=env.target_pos,
        fs=env.fs,
        sofa_path=env.sofa_path,
        lstn_pos=env.lstn_pos,
        device=env.device,
        dtype=env.dtype,
    )

    regularisation = second_moment_regularisation(
        n_samples=env.n_filter_taps, device=env.device, dtype=env.dtype
    )

    loss_dict = {
        "localisation": ListenerBasedLoss(loc_model, 1.0, env.device, env.dtype),
        #"colouration": ListenerBasedLoss(col_model, 1.0, env.device, env.dtype),
        #"uncertainty": ListenerBasedLoss(unc_model, 1.0, env.device, env.dtype),
        #"ipd": ListenerBasedLoss(ipd_model, 1.0e-1, env.device, env.dtype),    
    }

    nn_model = MultiSourceFIR(
        n_samples=env.n_filter_taps,
        n_sources=env.n_sources,
    ).to(device=env.device, dtype=env.dtype)
    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[0, 0] = 1.0

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

    return stereo_prop, loss_dict, nn_model, nn_optimiser, regularisation


def render_loudspeaker_output(stimulus, filters):
    stimulus = to_numpy(stimulus).astype(np.float64).flatten()
    filters = to_numpy(filters).astype(np.float64)

    if filters.ndim == 3 and filters.shape[1] == 1:
        filters = filters[:, 0, :]
    elif filters.ndim == 1:
        filters = filters[np.newaxis, :]

    if filters.ndim != 2:
        raise ValueError("Estimated filters must be shape [n_sources, n_taps].")

    speaker_signals = []
    for filt in filters:
        speaker_signals.append(causal_conv1d(stimulus, filt))

    speaker_signals = np.stack(speaker_signals, axis=0)
    peak = np.max(np.abs(speaker_signals))
    if peak > 0:
        speaker_signals /= peak * 1.05

    return speaker_signals


def write_stereo_wav(path, stereo_signal, fs):
    signal = np.asarray(stereo_signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] not in (1, 2):
        raise ValueError("stereo_signal must be shape [1|2, samples].")

    signal = signal.T
    signal = np.clip(signal, -1.0, 1.0)
    int_samples = (signal * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as f:
        f.setnchannels(int(int_samples.shape[1]))
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(int_samples.tobytes())


def open_pygame_player(speaker_signals, fs):

    # Convert to int16 stereo for pygame
    signal = np.clip(speaker_signals.T, -1.0, 1.0)
    int_samples = (signal * 32767.0).astype(np.int16)

    # Write to an in-memory WAV buffer (pygame needs WAV format)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(int_samples.shape[1])
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(int_samples.tobytes())
    buf.seek(0)

    pygame.init()
    pygame.mixer.init(frequency=fs, size=-16, channels=int_samples.shape[1])
    sound = pygame.mixer.Sound(buf)

    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Loudspeaker Player")
    font = pygame.font.SysFont(None, 36)

    print("Pygame window open — press SPACE to play, ESC or close to quit.")

    running = True
    while running:
        screen.fill((30, 30, 30))
        label = font.render("Press SPACE to play", True, (200, 200, 200))
        screen.blit(label, (80, 90))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sound.stop()
                    sound.play()

    pygame.quit()


def main():
    args = parse_args()
    env = build_env(args)

    print(f"Using device: {env.device}")
    set_seed(seed=env.seed)

    training_stimulus_data_loader, _ = get_stim_loader(
        env.fs, env.stimulus_batch_size, env.n_stimulus,
        stimulus_type=env.selected_stim, device=env.device, dtype=env.dtype,
    )
    training_listener_data_loader, _ = get_listener_loader(
        env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling
    )

    stereo_prop, loss_dict, nn_model, nn_optimiser, _ = initialise_models(env)

    trainer = Train(
        n_epoch=env.epochs,
        stim_loader=training_stimulus_data_loader,
        lstn_loader=training_listener_data_loader,
        nn=nn_model,
        prop=stereo_prop,
        loss_dict=loss_dict,
        optimiser=nn_optimiser,
        loggers=None,
    )

    start_time = time.time()
    trainer.train()
    print(f"Optimisation finished in {time.time() - start_time:.2f} seconds.")

    eval_lstn_pos, eval_perceptual_dict, eval_stimulus_data_loader, eval_stimulus_dataset = build_eval_objects(env)
    est_filters, estimation_dict, _ = trainer.evaluate(
        eval_lstn_pos=eval_lstn_pos,
        eval_stimulus_data_loader=eval_stimulus_data_loader,
        loss_dict=eval_perceptual_dict,
    )
    est_filters = to_numpy(est_filters)
    peak = np.max(np.abs(est_filters))
    if peak > 0:
        est_filters = est_filters / (peak / 0.9)

    stimulus = np.random.randn(48000)
    stimulus /= np.max(np.abs(stimulus)) / 0.9

    speaker_signals = render_loudspeaker_output(stimulus, est_filters)
    write_stereo_wav(args.output, speaker_signals, env.fs)
    print(f"Rendered loudspeaker output saved to {args.output}.")

    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    if args.play:
        open_pygame_player(speaker_signals, env.fs)

if __name__ == "__main__":
    main()