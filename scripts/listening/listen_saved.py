import io
import wave
import pygame
import argparse
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run optimisation and render estimated loudspeaker signals for listening."
    )
    parser.add_argument("--path", type=str)
    parser.add_argument("--stimulus", type=str, default="white_noise")
    parser.add_argument("--fs", type=int, default=48000)
   
    return parser.parse_args()


def causal_conv1d(input, weight):
    K = weight.shape[-1]
    pad_input = np.pad(input, (K-1, 0))
    out = np.convolve(pad_input, weight, mode="valid") 
    return out


def render_loudspeaker_output(stimulus, filters):
    stimulus = stimulus.astype(np.float64).flatten()
    filters = filters.astype(np.float64)

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
        
    if args.stimulus == "impulse":
        stimulus = np.zeros(48000)
        stimulus[0] = 1.0
    elif args.stimulus == "sine":
        stimulus = np.sin(2 * np.pi * 440 * np.arange(48000) / args.fs)
    elif args.stimulus == "white_noise":
        stimulus = np.random.randn(48000)
        stimulus /= np.max(np.abs(stimulus)) / 0.9
    else:
        print("Selected stimulus not available, using white noise")
        stimulus = np.random.randn(48000)
        stimulus /= np.max(np.abs(stimulus)) / 0.9

    # Load estimated filters
    filters_path = Path(args.path) / "artifacts" / "est_filters.npy"
    filters = np.load(filters_path)
    
    speaker_signals = render_loudspeaker_output(stimulus, filters)

    open_pygame_player(speaker_signals, args.fs)

if __name__ == "__main__":
    main()