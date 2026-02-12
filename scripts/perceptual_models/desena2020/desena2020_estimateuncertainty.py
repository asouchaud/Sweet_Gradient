import torch

def desena2020_estimateuncertainty(distance, Hmin = 0.8822):

    weightedLikelihood = distance['weightedLikelihood']
    n_angles = weightedLikelihood.shape[2]

    values = torch.arange(1,38, dtype=weightedLikelihood.dtype, device = weightedLikelihood.device).view(1,1,n_angles)
    cos = torch.cos(2*values*2*torch.pi/n_angles)
    sin = torch.sin(2*values*2*torch.pi/n_angles)

    alpha = torch.sum(weightedLikelihood*cos, dim = 2, keepdim=True)
    beta = torch.sum(weightedLikelihood*sin, dim = 2, keepdim=True)

    H = 1 - torch.sqrt(torch.pow(alpha,2) + torch.pow(beta,2))

    Hnorm = (H - Hmin)/(1-Hmin)

    return(Hnorm)