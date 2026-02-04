import numpy as np
import matplotlib.pyplot as plt

def graph_plot(est_filters, fs):

    plt.figure(1)
    plt.plot(est_filters[0], label='Left loudspeaker filter')
    plt.plot(est_filters[1], label='Right loudspeaker filter')
    plt.title('Estimated FIRs after training')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    n_fft = 4096
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    mag_fft_left = 20*np.log10(np.abs(np.fft.rfft(est_filters[0], n = n_fft)))
    mag_fft_right = 20*np.log10(np.abs(np.fft.rfft(est_filters[1], n = n_fft)))

    plt.figure(2)
    plt.semilogx(freqs, mag_fft_left, label='Left loudspeaker filter')
    plt.semilogx(freqs, mag_fft_right, label='Right loudspeaker filter')
    plt.title('Estimated FIRs after training (Frequency Domain)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()