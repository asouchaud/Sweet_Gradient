import numpy as np
import matplotlib.pyplot as plt

def graph_IC(est_filters):

    colors = np.arange(len(est_filters[0]))  # or est_filters[1]

    plt.figure(1)
    plt.scatter(
        est_filters[1],
        est_filters[0],
        c=colors,
        cmap='viridis',
        label='Left loudspeaker filter'
    )
    plt.colorbar(label='1/3 octave band index')
    plt.title('ICLD/ICTD pairs after training')
    plt.xlabel('ICTD [ms]')
    plt.ylabel('ICLD [dB]')
    plt.ylim(-20,20)
    plt.xlim(-1,1)
    plt.grid()
    plt.show()
