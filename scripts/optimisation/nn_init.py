import torch
import torch.nn as nn

def load_optimiser(nn, lr = 0.01): # Optimiser definition
    lr = 0.01
    optimiser = torch.optim.Adam(nn.parameters(), lr=lr)
    return optimiser

def load_NN(n_samples = 128, device = 'cuda', dtype = torch.float, reg = 1):

    if n_samples is None:
        NN = nn.Sequential(
            nn.Linear(1,1, bias=False) # Initialise NN with a single weight for ICLD 
        ).to(device=device, dtype=dtype)


    else:
        NN = nn.Sequential(
            nn.Linear(1,2*n_samples, bias=False) # Initialise NN with 2 times filter taps for both loudspeakers
        ).to(device=device, dtype=dtype)

        if reg: # Initialisation scheme
            with torch.no_grad():
                NN[0].weight.fill_(0.0)
                NN[0].weight[-1, 0] = 1.0
                NN[0].weight[n_samples-1, 0] = 1.0

    return NN