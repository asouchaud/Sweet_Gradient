import sofar as sf 
import torch
import numpy as np 
import os 
import scipy


def sofa_find(angle, sofaile, elv = 0):
    
    # Find index of closest angle in sofaile.SourcePosition
    sofa_angles = sofaile.SourcePosition[:, :2] % 360  # N_y x (az, elv)
    angle = np.atleast_1d(angle)  
    angles = np.concatenate((angle[:, None], elv*np.ones_like(angle[:, None])), axis=1) % 360  # N_x x (az, elv)
    diff = sofa_angles[:, None, :] - angles[None, :, :]  # N_y x N_x x (az, elv)
    dist = np.linalg.norm(diff, axis=-1)  # N_y x N_x
    idx = np.argmin(dist, axis=0)  # N_x

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