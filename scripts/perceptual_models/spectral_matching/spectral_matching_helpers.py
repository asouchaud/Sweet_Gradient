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
