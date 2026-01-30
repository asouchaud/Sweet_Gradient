import torch

from scripts.optimisation.curriculum import curriculum_weights
from scripts.optimisation.regularisation import select_regularistion
from scripts.optimisation.regularisation import rms_normalisation
from scripts.optimisation.regularisation import energy_normalisation

class Train:
    def __init__(self, n_epoch, nn, prop, perceptual_dict, optimiser, reg=0):

        self.n_epoch = n_epoch
        self.device = nn[0].weight.device
        self.dtype = nn[0].weight.dtype
        self.optimiser = optimiser
        self.n_samples = nn[-1].weight.shape[0] // 2
        self.reg = reg
        
        self.nn = nn
        self.prop = prop
        self.perceptual_dict = perceptual_dict
        self.regularisation = select_regularistion(reg, n_samples=self.n_samples, device=self.device, dtype=self.dtype)

    def train(self):
        for epoch in range(self.n_epoch):
        
            # Estimate FIRs 
            h = self.nn(torch.ones(1,1, device=self.device, dtype=self.dtype))
            h_l = h[:,:self.n_samples].unsqueeze(1)
            h_r = h[:,self.n_samples:].unsqueeze(1)
            h_full = torch.cat([h_l,h_r], dim=1)
            # h_full_normalised = energy_normalisation(h_full)


            # Propagate sound from speakers
            bin_stim = self.prop(h_full)

            # Get perceptual estimations and compute losses
            loss_dict = {}
            for entry in self.perceptual_dict.items():
                model = entry[1] # Get model
                model_target = model.target # Get target
                model_loss = model.loss # Get loss function

                model_estimation = model(bin_stim) # Get model estimation
                loss = model_loss(model_estimation, torch.tensor([model_target], device=self.device, dtype=self.dtype).expand_as(model_estimation)) # Calculate loss
                loss_dict[entry[0]] = loss / model.loss_norm # Store loss in dictionary
            
            # Compute regularisation loss
            if self.reg:
                loss_dict['regularisation_l'] = self.regularisation.compute_loss(h_l) / self.regularisation.loss_norm
                loss_dict['regularisation_r'] = self.regularisation.compute_loss(h_r) / self.regularisation.loss_norm

            # # Get curriculum weighted loss
            # weights, _ = curriculum_weights(epoch, self.n_epoch, self.optimiser, h_l, h_r, loss_dict['localization'], loss_dict['uncertainty'], loss_dict['coloration'])
            weights = {'localisation': 1.0, 'uncertainty': 1.0, 'coloration': 1.0}#, 'regularisation_l': 1.0, 'regularisation_r': 1.0} # Temporary fixed weights

            if self.reg:
                weights = {'localisation': 1.0, 'uncertainty': 1.0, 'coloration': 1.0, 'regularisation_l': 1.0, 'regularisation_r': 1.0} # Temporary fixed weights

            # Compute full loss
            full_loss = 0
            for key in loss_dict.keys():
                full_loss += weights.get(key, 0) * loss_dict[key]

            # Optimize
            self.optimiser.zero_grad()
            full_loss.backward()
            self.optimiser.step()
            
            # Print progress
            # print(f'Iter: {epoch} ¦¦ Loss: {loss.item()} ¦¦ Loc: {estimated_localization.mean().item()} ¦¦ Unc: {estimated_uncertainty.mean().item()} ¦¦ Color: {loss_coloration.item()}' )
            print(
            f"\rIter: {epoch:<4d} ¦¦ "
            f"Loss: {full_loss.item():<10.4f} ¦¦ ",
            # f"Loc: {estimated_localization.mean().item():<8.4f} ¦¦ "
            # f"Unc: {estimated_uncertainty.mean().item():<8.4f} ¦¦ "
            # f"Color: {loss_coloration.item():<8.4f}",
            end="",
            flush=True
            )
            print()

    def evaluate(self):
        # Evaluate final FIRs
        with torch.no_grad():
            h = self.nn(torch.ones(1,1, device=self.device, dtype=self.dtype))
            h_l = h[:,:self.n_samples].unsqueeze(1)
            h_r = h[:,self.n_samples:].unsqueeze(1)
            h_full = torch.cat([h_l,h_r], dim=1)

            # Propagate sound from speakers
            bin_stim = self.prop(h_full)

            estimation_dict = {}
            # Get perceptual estimations and compute losses
            for entry in self.perceptual_dict.items():
                model = entry[1] # Get model

                model_estimation = model(bin_stim) # Get model estimation
                estimation_dict[entry[0]] = model_estimation.squeeze().detach().cpu().numpy() # Store estimation in dictionary

        return h_full.squeeze().detach().cpu().numpy(), estimation_dict