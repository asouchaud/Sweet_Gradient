import sofar as sf 
import torch
import numpy as np 
import os 
import scipy


def get_IRs(idx, sofaile, device = 'cpu', dtype = torch.float64):
    # Get hrir
    assert isinstance(idx, type(np.array([0])))

    hrir = torch.tensor(sofaile.Data_IR[idx], device=device, dtype=dtype)

    return hrir

def sofa_find(angle, sofaile, elv=0):

    # Handle tensor
    if isinstance(angle, torch.Tensor):
        angle = angle.detach().cpu().numpy()

    # Convert int/float/array to numpy array, then ensure 1D
    angle = np.atleast_1d(np.asarray(angle, dtype=float))

    assert angle.ndim == 1, "Angle must be a scalar or 1D array"

    # Find index of closest angle in sofaile.SourcePosition
    sofa_angles = sofaile.SourcePosition[:, :2] % 360  # N_y x (az, elv)
    angles = np.concatenate((angle[:, None], elv * np.ones_like(angle[:, None])), axis=1) % 360  # N_x x (az, elv)
    diff = sofa_angles[:, None, :] - angles[None, :, :]  # N_y x N_x x (az, elv)
    dist = np.linalg.norm(diff, axis=-1)  # N_y x N_x
    idx = np.argmin(dist, axis=0)  # N_x

    assert np.allclose(sofaile.SourcePosition[idx, :2], angles, rtol=1)  # TODO: Set rtol to value that makes sense

    return idx


def get_hrirs_for_template(sofa_path, device = 'cpu', dtype = torch.float64):
    
    sofaile = sf.read_sofa(sofa_path) 
    
    angles = np.linspace(-90, 90, 37) 
    indices =  sofa_find(angles, sofaile)
    
    # Get IRs from SOFA File and Pad for Desired Output 
    hrirs = sofaile.Data_IR[indices.astype(int),:,:]
    hrirs = np.pad(hrirs, ((0,0),(0,0),(0,4096-hrirs.shape[2])), mode='constant') # Pad to 4096 samples for GTFB filtering
    hrirs = torch.tensor(hrirs, dtype=dtype, device=device)

    return hrirs


def near_field_target(target):
    
    theta = torch.deg2rad(torch.as_tensor(target))

    x = torch.tensor(1.732, device=theta.device, dtype=theta.dtype)
    y = torch.tan(theta) * x
    
    return torch.stack((x, y)) # Works only for loudspeakers at x = 1 



def pos2angle(target_pos, lstn_pos):

    assert type(target_pos) == torch.Tensor
    
    lstn_pos_tensor = torch.as_tensor(
    lstn_pos, 
    dtype=target_pos.dtype, 
    device=target_pos.device
    )

    vec_diff = target_pos - lstn_pos_tensor  # (N, 2)

    # Angle relative to (1, 0)
    theta = torch.rad2deg(torch.atan2(vec_diff[:, 1], vec_diff[:, 0]))

    return theta


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


def set_target(lstn_pos, target):
    ## TODO: Make this dependent on listener position?
    assert type(lstn_pos) == torch.Tensor

    if isinstance(target, (int, float)):
        target_tensor = torch.full((len(lstn_pos),), target, device=lstn_pos.device, dtype=lstn_pos.dtype)
    
    elif isinstance(target, torch.Tensor):
        if target.numel() == 1:
            target_tensor = torch.full((len(lstn_pos),), target.item(), device=lstn_pos.device, dtype=lstn_pos.dtype)
        else:
            target_tensor = target.to(device=lstn_pos.device, dtype=lstn_pos.dtype)
    
    else:
        raise ValueError("Target must be int, float or torch.Tensor")
    
    return target_tensor

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)