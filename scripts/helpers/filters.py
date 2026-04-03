import numpy as np
import scipy.signal


def bandpass_fir(x, fs, highpass_hz=50.0, lowpass_hz=20000.0, numtaps=33):
    """
    FIR bandpass filtering along the last axis.

    Expected input:
    - x: array-like of shape (..., n_samples). Any leading dimensions are preserved (e.g., channels, ears, sources).
    - fs: sampling rate in Hz.
    - highpass_hz / lowpass_hz: band edges in Hz.
    - numtaps: FIR length (number of taps).

    Output:
    - y: ndarray with same shape as x, filtered along the last axis using same-length convolution.
    """
    # Keep cutoff safely below Nyquist to avoid firwin complaining for edge cases.
    lowpass_hz = min(lowpass_hz, (fs / 2) * 0.999)
    h = scipy.signal.firwin(numtaps, cutoff=[highpass_hz, lowpass_hz], fs=fs, pass_zero=False)
    # Convolve each 1D vector along the last axis; "same" keeps the original signal length.
    return np.apply_along_axis(lambda v: scipy.signal.convolve(v, h, mode="same", method="fft"), axis=-1, arr=x)

