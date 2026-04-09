import numpy as np
import torch
import torch.nn.functional as F
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

def near_field_target(target):
    
    theta = torch.deg2rad(torch.as_tensor(target))

    x = torch.tensor(1.732, device=theta.device, dtype=theta.dtype)
    y = torch.tan(theta) * x
    

    return torch.stack((x, y)) # Works only for loudspeakers at x = 1 


def near_field_comp_angles(target_pos, lstn_pos_conf):
    
    # # Project into complex 
    # lstn_pos_conf_tensor = torch.tensor(lstn_pos_conf)
    # vec_diff = target_pos.expand_as(lstn_pos_conf_tensor) - lstn_pos_conf_tensor

    # vec_i = torch.tensor([vec_diff[:,0], 1j*vec_diff[:,1]])
    # # Find angle via complex plane angle
    # theta = torch.rad2deg(torch.angle(vec_i))

    lstn_pos_conf_tensor = torch.as_tensor(
    lstn_pos_conf, 
    dtype=target_pos.dtype, 
    device=target_pos.device
    )

    vec_diff = target_pos - lstn_pos_conf_tensor  # (N, 2)

    # Angle relative to (1, 0)
    theta = torch.rad2deg(torch.atan2(vec_diff[:, 1], vec_diff[:, 0]))

    return theta


def llado_get_hrirs(sofa_path, device = 'cpu', dtype = torch.float64):
    
    sofaile = sf.read_sofa(sofa_path) 
    
    angles = np.linspace(-90, 90, 37) 
    indices =  sofa_find(angles, sofaile)
    
    # Get IRs from SOFA File and Pad for Desired Output 
    hrirs = sofaile.Data_IR[indices.astype(int),:,:]
    hrirs = np.pad(hrirs, ((0,0),(0,0),(0,4096-hrirs.shape[2])), mode='constant') # Pad to 4096 samples for GTFB filtering
    hrirs = torch.tensor(hrirs, dtype=dtype, device=device)


    return hrirs

def llado_set_target(lstn_pos_conf, target_pos, device = 'cpu', dtype = torch.float64):

    target_pos = torch.as_tensor(target_pos, device=device, dtype=dtype)

    if isinstance(lstn_pos_conf, type(None)):
        target = torch.rad2deg(torch.atan2(target_pos[1], target_pos[0]))
    elif isinstance(lstn_pos_conf, list):
        target = near_field_comp_angles(target_pos, lstn_pos_conf)
    else:
        print("Wrong data type, defaulting to global target position angle")
        target = torch.rad2deg(torch.atan2(target_pos[1], target_pos[0]))

    return target_pos, target


def llado_set_norm(loss, spk_pos, device = 'cpu', dtype = torch.float64):
    
    spk_pos = np.array(spk_pos)

    angles = np.rad2deg(np.arctan2(spk_pos[:,1], spk_pos[:,0]))
    angles = np.sort(angles)

    # Compute circular differences (including wrap-around)
    diffs = np.diff(angles)
    wrap_diff = (angles[0] + 360) - angles[-1]
    diffs = np.append(diffs, wrap_diff)

    # Take smallest equivalent angle (important!)
    diffs = np.minimum(diffs, 360 - diffs)

    max_angle_diff = np.max(diffs)
    
    norm = loss(torch.tensor([max_angle_diff]), torch.tensor([0.0])) # Maximum possible loss for normalization

    return norm.to(device=device, dtype=dtype)