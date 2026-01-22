import torch

def curriculum_weights(i, epoch, optimizer, h_l, h_r, loss_localization, loss_uncertainty, loss_coloration):

    # Default weights
    w_loc, w_unc, w_col = 0, 0, 0

    if i == 0:
        desc = "Localization only"
        w_loc = 1
    elif i == 1:
        desc = "Uncertainty only"
        w_unc = 1
    elif i == 2:
        desc = "Coloration only"
        w_col = 1
    elif i == 3:
        desc = "Localization + Uncertainty (>300)"
        w_loc = 1
        if epoch >= 300:
            w_unc = 1
    elif i == 4:
        desc = "Localization + Coloration (>300)"
        w_loc = 1
        if epoch >= 300:
            w_col = 1
    elif i == 5:
        desc = "Uncertainty + Coloration (>300)"
        w_unc = 1
        if epoch >= 300:
            w_col = 1
    elif i == 6:
        desc = "Localization + Uncertainty(>300) + Coloration(>700)"
        if epoch < 300:
            w_loc = 1
        elif epoch < 700:
            w_loc, w_unc = 1, 1
        else:
            w_loc, w_unc, w_col = 1, 1, 1

    loss_regularization_l = torch.sum((reg_idx*h_l**2))
    loss_regularization_r = torch.sum((reg_idx*h_r**2))

    return (w_loc*loss_localization/(torch.tensor([3600], device=device, dtype=dtype)) 
            + w_unc*loss_uncertainty 
            + w_col*loss_coloration/(torch.tensor([15051.9082], device= device, dtype = dtype)) 
            + loss_regularization_l/(2*torch.tensor([9762.367], device=device, dtype=dtype)) 
            + loss_regularization_r/(2*torch.tensor([9762.367], device=device, dtype=dtype)), 
            desc)