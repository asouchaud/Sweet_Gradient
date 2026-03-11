import torch 
import numpy as np
import matplotlib.pyplot as plt

from scripts.perceptual_models.desena2020.desena2020_distance import desena2020_distance
from scripts.perceptual_models.desena2020.desena2020_estimateinterauralcues import desena2020_estimateinterauralcues
from scripts.perceptual_models.desena2020.desena2020_estimateuncertainty import desena2020_estimateuncertainty
from scripts.perceptual_models.desena2020.desena2020_loudnessweighting import desena2020_loudnessweighting  


def desena2020_Hmin(template):

    template_target = template.copy()
    template_template = template.copy()

    template_target['ild'] = template_target['ild'].unsqueeze(0)
    template_target['itd'] = template_target['itd'].unsqueeze(0)

    distance = desena2020_distance(template_target, template_template)

    distance = desena2020_loudnessweighting(template_target, distance)

    estimated_uncertainty = desena2020_estimateuncertainty(distance, template['azimuths'], 0)

    Hmin = torch.min(estimated_uncertainty)

    return(Hmin)