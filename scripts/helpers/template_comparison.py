import torch

USED_BAND_SLICE = slice(None)

def itd_ild_template_distance(target,template):

    target_itd = target['itd'].unsqueeze(2)
    target_ild = target['ild'].unsqueeze(2)
    
    device = target_itd.device
    dtype = target_itd.dtype

    template_itd = template['itd'].to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
    template_ild = template['ild'].to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    # Linear Weighting Over Frequency Bands
    weighting_function_itd_ild = torch.linspace(0, 1, 24, device=device, dtype=dtype)[USED_BAND_SLICE]

    ## Normalize Template and Target
    max_per_entry_itd = torch.max(torch.abs(template_itd), dim=2, keepdim=True).values # ABS SHOULD BE THERE, TESTING
    max_per_entry_ild = torch.max(torch.abs(template_ild), dim=2, keepdim=True).values

    template_itd_normalized = template_itd / (max_per_entry_itd + 1e-12)
    template_ild_normalized = template_ild / (max_per_entry_ild + 1e-8)

    target_itd_normalized = target_itd / (max_per_entry_itd + 1e-12)
    target_ild_normalized = target_ild / (max_per_entry_ild + 1e-8)
    
    # Calculate Distance Between Template and Target per Frequency Band
    p = 0.7
    distance_itd = torch.pow(torch.abs(target_itd_normalized - template_itd_normalized) +1e-8, p)
    distance_ild = torch.pow(torch.abs(target_ild_normalized - template_ild_normalized) +1e-8, p)

    # Weighted Sum of ILD and ITD Distances per Frequency Band
    distance_combined = torch.pow(distance_itd * (1 - weighting_function_itd_ild) + distance_ild * weighting_function_itd_ild, 1)
    # distance_combined = distance_itd + distance_ild
    
    # Turn Function into Probability Distribution
    distance_likelihood_f = torch.softmax(-distance_combined, dim=2)    

    ## Return
    
    distance = {
        "itd": distance_itd,
        "ild": distance_ild,
        "combined": distance_combined,
        "likelihood_f": distance_likelihood_f
    }

    return distance
