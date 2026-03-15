import torch
import torch.nn as nn

from scripts.optimisation.custom_modules import MultiSourceFIRModule


def load_optimiser(nn, lr = 0.01): # Optimiser definition
    optimiser = torch.optim.Adam(nn.parameters(), lr=lr)
    return optimiser

def load_NN(n_samples=128, device='cuda', dtype=torch.float, init=1):
    if n_samples is None:
        n_samples = 1

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
    elif init == 1:  # Impulses at both left and right speakers
        with torch.no_grad():
            for branch in NN.branches:
                branch.weight.fill_(0.0)
                branch.weight[0, 0] = 1.0
    elif init == 2:  # Impulse at left speaker, silent right speaker
        with torch.no_grad():
            NN[0].weight.fill_(0.0)
            NN[0].weight[0, 0] = 1.0
            NN[1].weight.fill_(0.0)

    return NN

def load_scheduler(optimiser, n_epochs):
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=n_epochs,     # total epochs
    eta_min=1e-5    # final LR
    )   
    
    return scheduler
