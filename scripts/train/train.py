import torch
import time

from scripts.optimisation.curriculum import gradnorm

from scripts.plotting.tensorboard_updates import TensorboardLogger

class Train:
    def __init__(self, n_epoch, nn, prop, perceptual_dict, loss_terms, optimiser, loggers=None, scheduler=None, gradnorm = 0, alpha = 1):

        self.n_epoch = n_epoch
        self.device = nn[0].weight.device
        self.dtype = nn[0].weight.dtype
        self.optimiser = optimiser
        self.n_samples = nn.n_samples
        self.n_sources = getattr(nn, 'n_sources', 2)
        self.gradnorm = gradnorm
        self.loss_terms = loss_terms

        weights = {
            "localisation": 1.0,
            "uncertainty": 1.0e1,
            "colouration": 1.0e-2,
        }
        self.loss_weights = torch.nn.ParameterDict({
            key: torch.nn.Parameter(
                torch.tensor(
                    1.0e-3 if key.startswith("regularisation_source_") else weights.get(key, 1.0),
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            for key in self.loss_terms.keys()
        })

        self.importance = None
        self.alpha = None
        if self.gradnorm:
            self.gradnorm_optimiser = torch.optim.Adam(self.loss_weights.parameters(), lr = 1e-2)
            base_importance = {
                                'localisation': 6.0,
                                'uncertainty': 1.0,
                                'colouration': 7.0,
                            }
            self.importance = {
                                **{key: base_importance.get(key, 1.0) for key in perceptual_dict.keys()},
                                **{key: 1.0 for key in self.loss_terms.keys() if key.startswith("regularisation_source_")}
                            }
            self.alpha = alpha

        self.nn = nn
        self.prop = prop
        self.perceptual_dict = perceptual_dict

        logger, saver = loggers[0], loggers[1]

        if logger is not None:
            if saver is not None:
                self.logger = TensorboardLogger(logger.run_dir / 'tensorboard' / saver.file_name[:-4], device = self.device, dtype = self.dtype)
            else: 
                self.logger = TensorboardLogger(logger.run_dir / 'tensorboard', device = self.device, dtype = self.dtype)
        else:
            self.logger = None

        self.scheduler = scheduler
        self.h_history = []

    def train(self):
        for epoch in range(self.n_epoch+1):
            time_epoch_start = time.time()

            # Estimate FIRs 
            h_full = self.nn(torch.ones(1,1, device=self.device, dtype=self.dtype))
            # h_full_normalised = energy_normalisation(h_full)
            with torch.no_grad():
                self.h_history.append(h_full.squeeze(0).detach().cpu().numpy())

            # Propagate sound from speakers
            bin_stim = self.prop(h_full)

            # Get perceptual estimations and compute losses
            loss_dict = {}
            est_dict = {}
            weighted_loss_dict = {}
            full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for key, loss_term in self.loss_terms.items():
                loss, estimation = loss_term(bin_stim=bin_stim, h_full=h_full)
                loss_dict[key] = loss
                weighted_loss = loss * self.loss_weights[key]
                weighted_loss_dict[key] = weighted_loss
                full_loss = full_loss + weighted_loss
                if estimation is not None:
                    est_dict[key] = estimation

            # Optimize Losses
            self.optimiser.zero_grad()
            full_loss.backward(retain_graph = True)
            self.optimiser.step()

            # Optimise weights using GradNorm (after network update)
            if self.gradnorm:
                gradnorm_loss, gradnorms, r = gradnorm(
                    self.nn,
                    loss_dict,
                    self.loss_weights,
                    self.importance,
                    alpha=self.alpha,
                )
                self.gradnorm_optimiser.zero_grad()
                gradnorm_loss.backward()
                self.gradnorm_optimiser.step()

                with torch.no_grad():
                    for p in self.loss_weights.values():
                        p.clamp_(min=0.0)
                    total_w = sum(p.item() for p in self.loss_weights.values())
                    N = sum(self.importance[k] for k in self.loss_weights.keys())
                    for p in self.loss_weights.values():
                        p.mul_(N / total_w)


            if self.scheduler is not None:
                self.scheduler.step()

            # Log training in Tensorboard
            if self.logger is not None:
                self.logger.log_losses(weighted_loss_dict, full_loss, epoch)
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
                f"Time: {time_epoch:.3f} seconds¦¦",
                end="",
                flush=True
                )
                print()
                


    def evaluate(self, perceptual_dict = None):

        if perceptual_dict is None:
            perceptual_dict = self.perceptual_dict
        # Evaluate final FIRs
        with torch.no_grad():
            h_full = self.nn(torch.ones(1,1, device=self.device, dtype=self.dtype))

            # Propagate sound from speakers
            bin_stim = self.prop(h_full)

            estimation_dict = {}
            # Get perceptual estimations and compute losses
            for entry in perceptual_dict.items():
                model = entry[1] # Get model

                model_estimation = model(bin_stim) # Get model estimation
                estimation_dict[entry[0]] = model_estimation.squeeze().detach().cpu().numpy() # Store estimation in dictionary

        return h_full.squeeze(0).detach().cpu().numpy(), estimation_dict, bin_stim.squeeze().detach().cpu().numpy()