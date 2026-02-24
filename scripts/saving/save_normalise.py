import numpy as np 

def normalise_max(est_filters):
    # Normalise filters to prevent clipping
    est_filters /= np.max(np.abs(est_filters)) / 0.9

    return est_filters