import torch
import torch.nn as nn

from scripts.optimisation.custom_modules import MultiSourceFIRModule


def load_optimiser(nn, lr = 0.01): # Optimiser definition
    optimiser = torch.optim.Adam(nn.parameters(), lr=lr)
    return optimiser

def load_NN(n_samples = 128, device = 'cuda', dtype = torch.float, init = 1):

    if n_samples is None:
        NN = nn.Sequential(
            nn.Linear(1,1, bias=False) # Initialise NN with a single weight for ICLD 
        ).to(device=device, dtype=dtype)


    else:
        NN = MultiSourceFIRModule(
            n_samples=n_samples,
            n_sources=2,
            layer_cls=nn.Linear,
            layer_kwargs={"bias": False},
        ).to(device=device, dtype=dtype)

        if init == 3:
            with torch.no_grad():
                for branch in NN.branches:
                    branch.weight.fill_(0.0)

        if init == 1:  # Initialisation scheme: impulses at both left and right speakers
            init_impulse = torch.zeros(n_samples, 1, device=device, dtype=dtype)
            init_impulse[0, 0] = 1.0
            with torch.no_grad():
                for branch in NN.branches:
                    branch.weight.copy_(init_impulse)

        if init == 2:  # Initialisation scheme: impulse at left speaker and silent right speaker
            init_impulse = torch.zeros(n_samples, 1, device=device, dtype=dtype)
            init_impulse[0, 0] = 1.0
            with torch.no_grad():
                NN[0].weight.copy_(init_impulse)
                NN[1].weight.fill_(0.0)

    return NN

def load_scheduler(optimiser, n_epochs):
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=n_epochs,     # total epochs
    eta_min=1e-5    # final LR
    )   
    
    return scheduler
