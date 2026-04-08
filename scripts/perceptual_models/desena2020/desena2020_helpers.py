import torch
import torch.nn.functional as F
import numpy as np
import sofar as sf



from scripts.setup.helpers import sofa_find

def softmax_st(z, dim = -1, tau=1.0):
    y_soft = F.softmax(z / tau, dim=dim)
    index = z.max(dim=dim, keepdim=True)[1]
    y_hard = torch.zeros_like(z).scatter_(dim, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft

def hilbert_transform(x):
    """
    Differentiable Hilbert transform for real input of shape (B, C, N).
    Returns the complex analytic signal of shape (B, C, N).
    """
    N = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)

    # Build the Hilbert filter mask
    h = torch.zeros(N, dtype=torch.float32, device=x.device)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    H = h.view(1, 1, N)  # Shape: [1, 1, N] to broadcast over B and C
    Xf_filtered = Xf * H

    x_analytic = torch.fft.ifft(Xf_filtered, dim=-1)
    return x_analytic  # Complex tensor: real + j*hilbert(real)

def rms(x, dim, keepdim = False):
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=keepdim))


def desena_get_hrirs(sofa_path, device = 'cpu', dtype = torch.float64):
    
    sofaile = sf.read_sofa(sofa_path) 
    
    angles = np.linspace(-90, 90, 37) 
    indices =  sofa_find(angles, sofaile)
    
    # Get IRs from SOFA File and Pad for Desired Output 
    hrirs = sofaile.Data_IR[indices.astype(int),:,:]
    hrirs = torch.tensor(hrirs, dtype=dtype, device=device)

    return hrirs



def desena_set_norm(loss, spk_pos, device = 'cpu', dtype = torch.float64):
    
    spk_pos = np.array(spk_pos)
    spk_angles = np.sort(np.rad2deg(np.arctan2(spk_pos[:,1], spk_pos[:,0]))%360)
    spk_angles_diff = spk_angles[1:] - spk_angles[:-1] # Get angle differences between adjacent speakers
    max_angle_diff = np.max(spk_angles_diff)%180 # Get maximum angle difference
    norm = loss(torch.tensor([max_angle_diff]), torch.tensor([0.0])) # Maximum possible loss for normalization

    return norm.to(device=device, dtype=dtype)