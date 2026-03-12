import torch
import time

from scripts.optimisation.curriculum import gradnorm
from scripts.optimisation.regularisation import select_regularistion
from scripts.optimisation.regularisation import rms_normalisation
from scripts.optimisation.regularisation import energy_normalisation

from scripts.plotting.tensorboard_updates import TensorboardLogger

class Train:
    def __init__(self, n_epoch, nn, prop, perceptual_dict, optimiser, reg=0, loggers=None, scheduler=None, gradnorm = 0, alpha = 1):

        self.n_epoch = n_epoch
        self.device = nn[0].weight.device
        self.dtype = nn[0].weight.dtype
        self.optimiser = optimiser
        self.n_samples = nn[-1].weight.shape[0] // 2
        self.reg = reg
        self.gradnorm = gradnorm

        if self.reg:
            self.loss_weights = torch.nn.ParameterDict({
                **{key: torch.nn.Parameter(torch.tensor(1.0, device=self.device, dtype=self.dtype))
                for key in perceptual_dict.keys()},
                'regularisation_l': torch.nn.Parameter(torch.tensor(1.0, device=self.device, dtype=self.dtype)),
                'regularisation_r': torch.nn.Parameter(torch.tensor(1.0, device=self.device, dtype=self.dtype)),
            })

        else:
            self.loss_weights = torch.nn.ParameterDict({
                key: torch.nn.Parameter(torch.tensor(1.0, device=self.device, dtype=self.dtype))
                for key in perceptual_dict.keys()
            })

        self.importance = None
        self.alpha = None
        if self.gradnorm:
            self.gradnorm_optimiser = torch.optim.Adam(self.loss_weights.parameters(), lr = 1e-2)
            self.importance = {
                                'localisation': 6.0,
                                'uncertainty': 1.0,
                                'colouration': 7.0,
                                'regularisation_l': 1.0,
                                'regularisation_r': 1.0
                            }
            self.alpha = alpha

        self.nn = nn
        self.prop = prop
        self.perceptual_dict = perceptual_dict
        self.regularisation = select_regularistion(reg, n_samples=self.n_samples, device=self.device, dtype=self.dtype)

        logger, saver = loggers[0], loggers[1]

        if logger is not None:
            if saver is not None:
                self.logger = TensorboardLogger(logger.run_dir / 'tensorboard' / saver.file_name[:-4], device = self.device, dtype = self.dtype)
            else: 
                self.logger = TensorboardLogger(logger.run_dir / 'tensorboard', device = self.device, dtype = self.dtype)
        else:
            self.logger = None

        self.scheduler = scheduler

    def train(self):
        for epoch in range(self.n_epoch):
            time_epoch_start = time.time()

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
            est_dict = {}
            for entry in self.perceptual_dict.items():
                model = entry[1] # Get model
                model_target = model.target # Get target
                model_loss = model.loss # Get loss function

                model_estimation = model(bin_stim) # Get model estimation
                loss = model_loss(model_estimation, torch.tensor(model_target, device=self.device, dtype=self.dtype).expand_as(model_estimation)) # Calculate loss
                loss_dict[entry[0]] = loss / model.loss_norm # Store loss in dictionary
                est_dict[entry[0]] = model_estimation # Store estimation in dictionary

            
            # Compute regularisation loss
            if self.reg:
                loss_dict['regularisation_l'] = self.regularisation.compute_loss(h_l) / self.regularisation.loss_norm
                loss_dict['regularisation_r'] = self.regularisation.compute_loss(h_r) / self.regularisation.loss_norm

            # Compute full loss
            full_loss = 0
            for key in loss_dict.keys():
                loss_dict[key] = loss_dict[key] * self.loss_weights[key]
                full_loss += loss_dict[key]

            # Optimize Losses
            self.optimiser.zero_grad()
            full_loss.backward(retain_graph = True)
            self.optimiser.step()

            # Optimise weights using GradNorm (after network update)
            if self.gradnorm:
<<<<<<< HEAD

                gradnorm_loss, gradnorms, r = gradnorm(self.nn, loss_dict, self.loss_weights, self.importance, alpha = self.alpha)
                
=======
                gradnorm_loss, gradnorms, r = gradnorm(
                    self.nn,
                    loss_dict,
                    self.loss_weights,
                    self.importance,
                    alpha=self.alpha,
                )
>>>>>>> Evaluation-plots
                self.gradnorm_optimiser.zero_grad()
                gradnorm_loss.backward()
                self.gradnorm_optimiser.step()

                with torch.no_grad():
                    for p in self.loss_weights.values():
                        p.clamp_(min=0.0)
<<<<<<< HEAD

                # Renormalize weights to sum = number of losses
=======
>>>>>>> Evaluation-plots
                    total_w = sum(p.item() for p in self.loss_weights.values())
                    N = sum(self.importance[k] for k in self.loss_weights.keys())
                    for p in self.loss_weights.values():
                        p.mul_(N / total_w)


            if self.scheduler is not None:
                self.scheduler.step()

            # Log training in Tensorboard
            if self.logger is not None:
                self.logger.log_losses(loss_dict, full_loss, epoch)
                self.logger.log_h_full(h_full, epoch)

                if 'localisation' in self.perceptual_dict or 'uncertainty' in self.perceptual_dict:
                    self.logger.log_bin_cues(bin_stim, epoch)

                if 'localisation' in self.perceptual_dict:
                    if self.perceptual_dict['localisation'].lstn_pos_conf is not None:
                        self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                   self.prop.spk_pos, 
                                                   est_dict['localisation'], 
                                                   est_dict['uncertainty'] if 'uncertainty' in self.perceptual_dict else None, 
                                                   est_dict['coloration'] if 'coloration' in self.perceptual_dict else None, 
                                                   epoch, 
                                                   loc_target=self.perceptual_dict['localisation'].target[0])
                    else:
                        self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                   self.prop.spk_pos, 
                                                   est_dict['localisation'], 
                                                   est_dict['uncertainty'] if 'uncertainty' in self.perceptual_dict else None, 
                                                   est_dict['coloration'] if 'coloration' in self.perceptual_dict else None, 
                                                   epoch)
                if self.gradnorm:
                    self.logger.log_weights(weight_dict=self.loss_weights, epoch=epoch)
                    self.logger.log_gradnorms(gradnorms = gradnorms, epoch=epoch)
                    self.logger.log_r(r,epoch=epoch)
            
            time_epoch = time.time() - time_epoch_start
            # Print progress
            if epoch % 50 == 0:
                print(
                f"\rIter: {epoch+1:<4d} ¦¦ "
                f"Loss: {full_loss.item():<10.5f} ¦¦ ",
                # f"Time: {time_epoch:.3f} seconds¦¦",
                end="",
                flush=True
                )
                print()
                pass

        ##### SAVE EVERYTHING AFTER FINAL BACKPROPAGATION ####
        with torch.no_grad():
            h = self.nn(torch.ones(1,1, device=self.device, dtype=self.dtype))
            h_l = h[:,:self.n_samples].unsqueeze(1)
            h_r = h[:,self.n_samples:].unsqueeze(1)
            h_full = torch.cat([h_l,h_r], dim=1)

            bin_stim = self.prop(h_full)

            # Get perceptual estimations and compute losses
            loss_dict = {}
            for entry in self.perceptual_dict.items():
                model = entry[1] # Get model
                model_target = model.target # Get target
                model_loss = model.loss # Get loss function

<<<<<<< HEAD
        # Get perceptual estimations and compute losses
        loss_dict = {}
        est_dict = {}
        for entry in self.perceptual_dict.items():
            model = entry[1] # Get model
            model_target = model.target # Get target
            model_loss = model.loss # Get loss function

            model_estimation = model(bin_stim) # Get model estimation
            loss = model_loss(model_estimation, torch.tensor(model_target, device=self.device, dtype=self.dtype).expand_as(model_estimation)) # Calculate loss
            loss_dict[entry[0]] = loss / model.loss_norm # Store loss in dictionary
            est_dict[entry[0]] = model_estimation # Store estimation in dictionary

=======
                model_estimation = model(bin_stim) # Get model estimation
                loss = model_loss(model_estimation, torch.tensor(model_target, device=self.device, dtype=self.dtype).expand_as(model_estimation)) # Calculate loss
                loss_dict[entry[0]] = loss / model.loss_norm # Store loss in dictionary

            # Compute regularisation loss
            if self.reg:
                loss_dict['regularisation_l'] = self.regularisation.compute_loss(h_l) / self.regularisation.loss_norm
                loss_dict['regularisation_r'] = self.regularisation.compute_loss(h_r) / self.regularisation.loss_norm
>>>>>>> Evaluation-plots

            # Log training in Tensorboard
                if self.logger is not None:
                    self.logger.log_losses(loss_dict, full_loss, epoch+1)
                    self.logger.log_h_full(h_full, epoch+1)

<<<<<<< HEAD
        # Log training in Tensorboard
        if self.logger is not None:
            self.logger.log_losses(loss_dict, full_loss, epoch+1)
            self.logger.log_h_full(h_full, epoch+1)

            if 'localisation' in self.perceptual_dict or 'uncertainty' in self.perceptual_dict:
                self.logger.log_bin_cues(bin_stim, epoch+1)

            if 'localisation' in self.perceptual_dict:
                if self.perceptual_dict['localisation'].lstn_pos_conf is not None:
                    self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                self.prop.spk_pos, 
                                                est_dict['localisation'], 
                                                est_dict['uncertainty'] if 'uncertainty' in self.perceptual_dict else None, 
                                                est_dict['coloration'] if 'coloration' in self.perceptual_dict else None, 
                                                epoch+1, 
                                                loc_target=self.perceptual_dict['localisation'].target[0])
                else:
                    self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                self.prop.spk_pos, 
                                                est_dict['localisation'], 
                                                est_dict['uncertainty'] if 'uncertainty' in self.perceptual_dict else None, 
                                                est_dict['coloration'] if 'coloration' in self.perceptual_dict else None, 
                                                epoch+1)
            if self.gradnorm:
                self.logger.log_weights(weight_dict=self.loss_weights, epoch=epoch+1)
                self.logger.log_gradnorms(gradnorms = gradnorms, epoch=epoch+1)
                self.logger.log_r(r,epoch=epoch+1)
=======
                    if 'localisation' in self.perceptual_dict or 'uncertainty' in self.perceptual_dict:
                        self.logger.log_bin_cues(bin_stim, epoch+1)

                    if 'localisation' in self.perceptual_dict:
                        if self.perceptual_dict['localisation'].lstn_pos_conf is not None:
                            self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                    self.prop.spk_pos, 
                                                    est_dict['localisation'], 
                                                    est_dict['uncertainty'] if 'uncertainty' in self.perceptual_dict else None, 
                                                    est_dict['coloration'] if 'coloration' in self.perceptual_dict else None, 
                                                    epoch+1, 
                                                    loc_target=self.perceptual_dict['localisation'].target[0])
                        else:
                            self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                    self.prop.spk_pos, 
                                                    est_dict['localisation'], 
                                                    est_dict['uncertainty'] if 'uncertainty' in self.perceptual_dict else None, 
                                                    est_dict['coloration'] if 'coloration' in self.perceptual_dict else None, 
                                                    epoch+1)
                    if self.gradnorm:
                        self.logger.log_weights(weight_dict=self.loss_weights, epoch=epoch+1)
                        self.logger.log_gradnorms(gradnorms = gradnorms, epoch=epoch+1)
                        self.logger.log_r(r,epoch=epoch+1)
>>>>>>> Evaluation-plots


    def evaluate(self, perceptual_dict = None):

        if perceptual_dict is None:
            perceptual_dict = self.perceptual_dict
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
            for entry in perceptual_dict.items():
                model = entry[1] # Get model

                model_estimation = model(bin_stim) # Get model estimation
                estimation_dict[entry[0]] = model_estimation.squeeze().detach().cpu().numpy() # Store estimation in dictionary

        return h_full.squeeze().detach().cpu().numpy(), estimation_dict, bin_stim.squeeze().detach().cpu().numpy()