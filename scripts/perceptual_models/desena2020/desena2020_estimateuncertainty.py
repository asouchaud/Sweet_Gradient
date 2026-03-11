import torch

def desena2020_estimateuncertainty(distance, natural_angles, Hmin = 0.8822):

    weightedLikelihood = distance['weightedLikelihood']
    n_angles = weightedLikelihood.shape[2]

    # values = torch.arange(0,37, dtype=weightedLikelihood.dtype, device = weightedLikelihood.device).view(1,1,n_angles)
    # cos = torch. 
    cos = torch.cos(torch.deg2rad(natural_angles)*2).view(1,1,n_angles)
    sin = torch.sin(torch.deg2rad(natural_angles)*2).view(1,1,n_angles)

    alpha = torch.sum(weightedLikelihood*cos, dim = 2, keepdim=True)
    beta = torch.sum(weightedLikelihood*sin, dim = 2, keepdim=True)

    H = 1 - torch.sqrt(torch.pow(alpha,2) + torch.pow(beta,2))

    Hnorm = (H - Hmin)/(1-Hmin)

    return(Hnorm)
