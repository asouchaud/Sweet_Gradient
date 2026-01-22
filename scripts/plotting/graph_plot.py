import torch
import numpy as np
import matplotlib.pyplot as plt

def graph_plot(est_filters, fs):

    plt.figure(1)
    plt.plot(est_filters[0,:], label='Left loudspeaker filter')
    plt.plot(est_filters[1,:], label='Right loudspeaker filter')
    plt.title('Estimated FIRs after training')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    freqs = torch.fft.rfftfreq(est_filters.shape[1], 1/fs)
    mag_fft_left = 20*torch.log10(torch.abs(torch.fft.rfft(torch.tensor(est_filters[0,:]))))
    mag_fft_right = 20*torch.log10(torch.abs(torch.fft.rfft(torch.tensor(est_filters[1,:]))))

    plt.figure(2)
    plt.plot(freqs, mag_fft_left, label='Left loudspeaker filter')
    plt.plot(freqs, mag_fft_right, label='Right loudspeaker filter')
    plt.title('Estimated FIRs after training (Frequency Domain)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()