import torch
import torch.nn.functional as F

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

    x = torch.tensor(1, device=theta.device, dtype=theta.dtype)
    y = torch.tan(theta)
    

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

    return theta % 360


