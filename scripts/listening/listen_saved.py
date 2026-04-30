import pygame
import argparse
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
try:
    import sounddevice as sd
except ImportError:
    sd = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run optimisation and render estimated loudspeaker signals for listening."
    )
    parser.add_argument("--path", type=str)
    parser.add_argument("--stimulus", type=str, default="white_noise")
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="sounddevice output device index. Defaults to system default output.",
    )
    parser.add_argument(
        "--allow-downmix",
        action="store_true",
        help="Allow automatic downmix when output device has fewer channels.",
    )
    parser.add_argument(
        "--channel-test",
        action="store_true",
        help="Before interactive playback, play a short test tone one channel at a time.",
    )
   
    return parser.parse_args()


def causal_conv1d(input, weight):
    K = weight.shape[-1]
    pad_input = np.pad(input, (K-1, 0))
    out = np.convolve(pad_input, weight, mode="valid") 
    return out


def normalise_filters_shape(filters):
    """Return filters as shape [n_sources, n_taps]."""
    filters = np.asarray(filters, dtype=np.float64)
    if filters.ndim == 1:
        return filters[np.newaxis, :]
    if filters.ndim == 2:
        return filters
    if filters.ndim == 3:
        # Common save formats: [n_sources, 1, n_taps] or [1, n_sources, n_taps]
        if filters.shape[1] == 1:
            return filters[:, 0, :]
        if filters.shape[0] == 1:
            return filters[0, :, :]
    raise ValueError(
        f"Estimated filters shape {filters.shape} is unsupported. "
        "Expected [n_sources, n_taps] (or equivalent singleton-expanded layout)."
    )


def render_loudspeaker_output(stimulus, filters):
    stimulus = stimulus.astype(np.float64).flatten()
    filters = normalise_filters_shape(filters)

    speaker_signals = []
    for filt in filters:
        speaker_signals.append(causal_conv1d(stimulus, filt))

    speaker_signals = np.stack(speaker_signals, axis=0)
    peak = np.max(np.abs(speaker_signals))
    if peak > 0:
        speaker_signals /= peak * 1.05

    return speaker_signals


def _get_output_device_info(output_device):
    def _available_output_devices_text():
        devices = sd.query_devices()
        output_rows = []
        for idx, info in enumerate(devices):
            max_out = int(info.get("max_output_channels", 0))
            if max_out < 1:
                continue
            output_rows.append(f"  {idx}: {info.get('name', 'Unknown')} (max_out={max_out})")
        if not output_rows:
            return "  <none>"
        return "\n".join(output_rows)

    if output_device is not None:
        device_idx = int(output_device)
        try:
            device_info = sd.query_devices(device_idx, "output")
        except ValueError as exc:
            raise ValueError(
                f"Device index {device_idx} is not a valid output device.\n"
                "Available output devices are:\n"
                f"{_available_output_devices_text()}"
            ) from exc
        return device_idx, device_info

    # Prefer MOTU hardware automatically when available.
    all_devices = sd.query_devices()
    for idx, info in enumerate(all_devices):
        if int(info.get("max_output_channels", 0)) < 1:
            continue
        name = str(info.get("name", "")).lower()
        if "motu" in name:
            return idx, sd.query_devices(idx, "output")

    default_out = sd.default.device[1]
    if default_out is None or int(default_out) < 0:
        raise RuntimeError(
            "No default output device found. Pass --output-device with a valid device index."
        )
    device_idx = int(default_out)
    device_info = sd.query_devices(device_idx, "output")
    return device_idx, device_info


def _downmix_to_channels(signal, target_channels):
    """Downmix [samples, n_channels] to [samples, target_channels]."""
    if target_channels <= 0:
        raise ValueError("target_channels must be >= 1")
    if signal.shape[1] == target_channels:
        return signal

    out = np.zeros((signal.shape[0], target_channels), dtype=signal.dtype)
    for ch in range(target_channels):
        grouped = signal[:, ch::target_channels]
        out[:, ch] = np.mean(grouped, axis=1)
    return out


def _prepare_signal_for_device(signal, max_device_channels, allow_downmix):
    requested_channels = int(signal.shape[1])
    if requested_channels <= max_device_channels:
        return signal, False
    if not allow_downmix:
        raise RuntimeError(
            f"Requested {requested_channels} channels, but output device supports only {max_device_channels}. "
            "Use --output-device to select another device, or pass --allow-downmix."
        )
    return _downmix_to_channels(signal, max_device_channels), True


def _prepare_audio_for_sounddevice(signal, target_peak=0.05):
    """Validate and normalise audio to safe float32 range for sounddevice."""
    audio = np.asarray(signal)
    if audio.ndim != 2:
        raise ValueError(
            f"Expected audio shape [samples, channels], got shape {audio.shape}."
        )
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("Audio buffer is empty.")

    if not np.all(np.isfinite(audio)):
        bad_count = int(np.size(audio) - np.count_nonzero(np.isfinite(audio)))
        raise ValueError(f"Audio contains {bad_count} non-finite samples (NaN/Inf).")

    audio = audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        print(f"Warning: clipping risk detected (peak={peak:.4f}). Renormalising to {target_peak:.2f}.")
        audio *= (target_peak / peak)
    elif peak > target_peak and peak > 0.0:
        # Keep a small headroom margin near full scale.
        audio *= (target_peak / peak)

    audio = np.clip(audio, -1.0, 1.0)
    return np.ascontiguousarray(audio, dtype=np.float32)


def run_channel_test(n_channels, fs, device_idx, screen, font):
    """Interactive channel test: SPACE advances and plays current channel."""
    test_duration_s = 0.35
    test_freq_hz = 440.0
    n_samples = max(1, int(round(test_duration_s * fs)))
    t = np.arange(n_samples, dtype=np.float32) / float(fs)
    tone = (0.2 * np.sin(2.0 * np.pi * test_freq_hz * t)).astype(np.float32)

    print("Channel test mode: SPACE = next channel, ENTER = done, ESC = quit.")
    current_channel = 0
    running = True
    while running:
        screen.fill((25, 25, 25))
        line1 = font.render("Channel test mode", True, (220, 220, 220))
        line2 = font.render(
            f"Current channel: {current_channel + 1}/{n_channels}", True, (220, 220, 220)
        )
        line3 = font.render("SPACE: play + next channel", True, (180, 180, 180))
        line4 = font.render("ENTER: finish test   ESC: quit script", True, (180, 180, 180))
        screen.blit(line1, (25, 65))
        screen.blit(line2, (25, 95))
        screen.blit(line3, (25, 125))
        screen.blit(line4, (25, 155))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sd.stop()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sd.stop()
                    return False
                if event.key == pygame.K_RETURN:
                    sd.stop()
                    return True
                if event.key == pygame.K_SPACE:
                    test_sig = np.zeros((n_samples, n_channels), dtype=np.float32)
                    test_sig[:, current_channel] = tone
                    test_sig = _prepare_audio_for_sounddevice(test_sig)
                    print(f"  Testing channel {current_channel + 1}/{n_channels}")
                    sd.stop()
                    sd.play(test_sig, samplerate=fs, device=device_idx, blocking=False)
                    current_channel = (current_channel + 1) % n_channels

    sd.stop()
    return True



def play_loudspeaker_output(
    speaker_signals, fs, output_device=None, allow_downmix=False, channel_test=False
):
    """Interactive player for any channel count (SPACE replay, ESC quit)."""
    if sd is None:
        raise RuntimeError(
            "Playback requires sounddevice. Install it with: pip install sounddevice"
        )
    device_idx, device_info = _get_output_device_info(output_device)
    max_device_channels = int(device_info["max_output_channels"])
    if max_device_channels < 1:
        raise RuntimeError(f"Selected output device {device_idx} has no output channels.")

    signal = _prepare_audio_for_sounddevice(speaker_signals.T)
    signal, was_downmixed = _prepare_signal_for_device(
        signal, max_device_channels=max_device_channels, allow_downmix=allow_downmix
    )
    signal = _prepare_audio_for_sounddevice(signal)
    n_channels = int(signal.shape[1])

    pygame.init()
    screen = pygame.display.set_mode((560, 220))
    pygame.display.set_caption("Loudspeaker Player")
    font = pygame.font.SysFont(None, 30)
    if was_downmixed:
        print(
            f"Downmixed to {n_channels} channels for device {device_idx} ('{device_info['name']}')."
        )
    if channel_test:
        keep_running = run_channel_test(
            n_channels=n_channels, fs=fs, device_idx=device_idx, screen=screen, font=font
        )
        if not keep_running:
            sd.stop()
            pygame.quit()
            return
    print(
        f"Sounddevice player ready ({n_channels} channels, device {device_idx}: {device_info['name']}) "
        "— SPACE toggles play/pause, ESC or close quits."
    )

    playback_state = {
        "cursor": 0,
        "playing": False,
    }
    n_total_samples = int(signal.shape[0])

    def _callback(outdata, frames, time, status):
        del time
        if status:
            print(f"Sounddevice status: {status}")
        if not playback_state["playing"]:
            outdata.fill(0.0)
            return

        filled = 0
        while filled < frames:
            start = playback_state["cursor"]
            remaining_in_buffer = n_total_samples - start
            n_copy = min(frames - filled, remaining_in_buffer)
            outdata[filled:filled + n_copy] = signal[start:start + n_copy]
            filled += n_copy
            playback_state["cursor"] = (start + n_copy) % n_total_samples

    stream = None

    running = True
    while running:
        screen.fill((30, 30, 30))
        if playback_state["playing"]:
            line1 = font.render("SPACE: pause", True, (220, 220, 220))
        elif playback_state["cursor"] > 0:
            line1 = font.render("SPACE: resume", True, (220, 220, 220))
        else:
            line1 = font.render("SPACE: play", True, (220, 220, 220))
        line2 = font.render("ESC: quit", True, (180, 180, 180))
        screen.blit(line1, (25, 85))
        screen.blit(line2, (25, 120))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if playback_state["playing"]:
                        playback_state["playing"] = False
                        if stream is not None and stream.active:
                            stream.stop()
                    else:
                        playback_state["playing"] = True
                        try:
                            if stream is None:
                                stream = sd.OutputStream(
                                    samplerate=fs,
                                    channels=n_channels,
                                    dtype="float32",
                                    device=device_idx,
                                    callback=_callback,
                                )
                            if not stream.active:
                                stream.start()
                        except sd.PortAudioError as exc:
                            playback_state["playing"] = False
                            raise RuntimeError(
                                f"Could not open output device {device_idx} ('{device_info['name']}') "
                                f"for {n_channels} channels at {fs} Hz: {exc}"
                            ) from exc

    if stream is not None:
        stream.stop()
        stream.close()
    sd.stop()
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
    elif args.stimulus == "finger_snaps":
        wav_path = Path(__file__).resolve().parents[2] / "data" / "stimulus" / "finger snaps.wav"
        src_fs, stimulus = wavfile.read(wav_path)
        stimulus = np.asarray(stimulus, dtype=np.float64)
        if stimulus.ndim > 1:
            stimulus = np.mean(stimulus, axis=1)
        if src_fs != args.fs:
            target_len = int(round(len(stimulus) * args.fs / src_fs))
            stimulus = signal.resample(stimulus, target_len)
        peak = np.max(np.abs(stimulus))
        if peak > 0:
            stimulus /= peak / 0.9
    else:
        print("Selected stimulus not available, using white noise")
        stimulus = np.random.randn(48000)
        stimulus /= np.max(np.abs(stimulus)) / 0.9

    # Load estimated filters
    filters_path = Path(args.path) / "artifacts" / "est_filters.npy"
    filters = np.load(filters_path)
    
    speaker_signals = render_loudspeaker_output(stimulus, filters)
    play_loudspeaker_output(
        speaker_signals,
        args.fs,
        output_device=args.output_device,
        allow_downmix=args.allow_downmix,
        channel_test=args.channel_test,
    )

if __name__ == "__main__":
    main()