import torch

from scripts.perceptual_models.llado2025.DEPRACTED_llado2025_helpers import softmax_st

def llado2025_estimatelocalisation_torch(distance, template, tau_L=1):

    # Find most likely direction of arrival per frequency band
    estimated_localisation_f = torch.sum(torch.mul(softmax_st(distance['likelihood_f'],dim=2, tau = tau_L),template['azimuths']), dim=2) # Softmax straight-through estimator
    # estimated_localisation_f = torch.sum(torch.mul(torch.softmax(distance['likelihood_f']/tau_L,dim=2),template['azimuths']), dim=2) # Basic Softmax
 
    # Average most likely direction of arrival over frequency band
    estimated_localisation = torch.mean(estimated_localisation_f, dim=2)

    return estimated_localisation