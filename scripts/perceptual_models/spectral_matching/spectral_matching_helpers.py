import torch

def near_field_target(target):
    # DEFINED ONLY FOR LOUDSPEAKER LOCATION AT (1,1) AND (1,-1) AND SWEET SPOT AT (-0.732,0)

    # y = torch.sin(torch.deg2rad(torch.tensor(target))) * 2 
    # x = torch.sin(torch.deg2rad(torch.tensor(target))) * 2

    # return torch.tensor([x.item(),y.item()])
    
    theta = torch.deg2rad(torch.as_tensor(target))
    
    x = torch.tensor(1.732, device=theta.device, dtype=theta.dtype)
    y = torch.tan(theta)*x
    

    return torch.stack((x, y)) 


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


def log_spectral_distortion(x, y, n_fft=None, eps=1e-8):
    """
    Compute LSD between two time signals.

    Args:
        x, y: tensors of shape [a, b, c]
        n_fft: FFT size (default: c)

    Returns:
        LSD tensor of shape [a]
    """

    if n_fft is None:
        n_fft = x.shape[-1]

    # FFT along time dimension
    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    Y = torch.fft.rfft(y, n=n_fft, dim=-1)

    # Magnitude
    X_mag = torch.abs(X)
    Y_mag = torch.abs(Y)

    # Log magnitude (dB)
    log_X = 20 * torch.log10(X_mag + eps)
    log_Y = 20 * torch.log10(Y_mag + eps)

    # LSD over frequency bins
    lsd_freq = torch.sqrt(torch.mean((log_X - log_Y) ** 2, dim=-1))  # [a, b]

    # Average across b
    lsd = lsd_freq.mean(dim=-1)  # [a]

    return lsd