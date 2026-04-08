import torch 
import numpy as np 
import os 
import scipy

def math_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.nn.functional.conv1d(input, torch.flip(weight, dims = [-1]), bias, stride, padding, dilation, groups)

def true_conv1d(input, weight):
    K = weight.shape[-1]
    pad_input = torch.nn.functional.pad(input, (K-1,0))
    out = torch.nn.functional.conv1d(pad_input, torch.flip(weight, [-1]))
    
    return out

def l3_loss(input, target, reduction="mean"):
    loss = torch.abs(input - target) ** 3
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
    
def sofa_find(angle, sofaile):
    # Find index of closest angle in sofaile.SourcePosition
    sofa_angles = sofaile.SourcePosition[:,:2] # N_y x (az, elv)
    angles = np.concat((angle[:, None], np.zeros_like(angle[:, None])), axis=1) # N_x x (az, elv)
    diff = sofa_angles[:,None,:] - angles[None,:,:] # N_y x N_x x (az, elv)
    dist = np.linalg.norm(diff, axis=-1) # N_y x N_x
    idx = np.argmin(dist, axis=0) # N_x
    return idx
    

def load_gtfb(fs, device = 'cpu', dtype = torch.float64):
    # Load Gammatone Filter Bank Impulse Responses
    script_dir = os.path.dirname(__file__)
    gtfb_path = os.path.join(script_dir, "..","..","data", "impulses","GTFB", "GTFB_4096_fs_48000.mat")
    gtfb_path = os.path.normpath(gtfb_path)

    MATLAB_GTFB_dict = scipy.io.loadmat(gtfb_path)
    gtfb = torch.tensor(np.real(MATLAB_GTFB_dict['gtfb'].T), dtype=dtype, device=device).unsqueeze(1)

    if fs != 48000:
        # Resample GTFB to desired sampling frequency
        num_samples = int(gtfb.shape[-1] * fs / 48000)
        gtfb = torch.nn.functional.interpolate(gtfb, size=num_samples, mode='linear', align_corners=False)

    return gtfb