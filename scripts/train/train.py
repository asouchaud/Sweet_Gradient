import torch
import time

from scripts.plotting.tensorboard_updates import TensorboardLogger

# Listener-based losses operate on the propagated binaural scene and a perceptual model
# (e.g., localisation, uncertainty, colouration). 
class ListenerBasedLoss:
    def __init__(self, model, weight, device, dtype):
        self.model = model
        self.weight = weight
        self.device = device
        self.dtype = dtype

    def compute_loss(self, bin_stim, h_full):  # noqa: ARG002 - kept for uniform interface
        """
        Args:
            bin_stim: Tensor shaped (batch, n_listeners, 2, n_samples), propagated binaural signals, 
            and 2 is the number of ears.
            h_full: Tensor shaped (batch, n_sources, n_samples), source filters (unused here).

        Why include an unused input:
            Train uses one generic loop over `loss_dict` and always calls
            `compute_loss(bin_stim=..., h_full=...)` regardless of loss type.
            Keeping the same method signature across ListenerBasedLoss and
            SourceBasedLoss is intentional polymorphism: it avoids branching
            logic in Train and makes it easy to add new loss classes later.
            For listener-based losses, `h_full` is accepted for API consistency.

        Returns:
            loss: Scalar tensor.
            model_estimation: Tensor output of the perceptual model, shape depends on the model.
        """
        model_estimation = self.model(bin_stim)
        target = torch.as_tensor(self.model.target, device=self.device, dtype=self.dtype).expand_as(model_estimation)
        loss = self.model.loss(model_estimation, target) / self.model.loss_norm
        return loss, model_estimation


# Source-based losses regularise individual source filters directly from h_full.
# They do not expose a perceptual model, so Train treats them as optimization-only
# objectives (no perceptual estimation output needed for evaluate/plotting).
class SourceBasedLoss:
    def __init__(self, regulariser, source_idx, weight):
        self.model = None
        self.regulariser = regulariser
        self.source_idx = source_idx
        self.weight = weight

    def compute_loss(self, bin_stim, h_full):  # noqa: ARG002 - kept for uniform interface
        """
        Args:
            bin_stim: Tensor shaped (batch, n_listeners, 2, n_samples), propagated binaural signals, 
            and 2 is the number of ears (unused here).
            h_full: Tensor shaped (batch, n_sources, n_samples), source filters.

        Why include an unused input:
            This class shares the same `compute_loss` signature as
            ListenerBasedLoss so Train can call all losses uniformly.
            Here only `h_full` is needed, while `bin_stim` is accepted for
            API consistency and future extensibility.

        Returns:
            loss: Scalar tensor for the selected source filter.
            estimation: None (source-based losses do not produce perceptual estimates).
        """
        source_filter = h_full[:, self.source_idx:self.source_idx + 1, :]
        loss = self.regulariser.compute_loss(source_filter) / self.regulariser.loss_norm
        return loss, None

class Train:
    def __init__(self, n_epoch, nn, prop, loss_dict, optimiser, loggers=None, scheduler=None, alpha = 1):

        self.n_epoch = n_epoch
        self.device = nn[0].weight.device
        self.dtype = nn[0].weight.dtype
        self.optimiser = optimiser
        self.n_samples = nn.n_samples
        self.n_sources = getattr(nn, 'n_sources', 2)
        self.loss_dict = loss_dict
        self.loss_weights = {
            key: torch.as_tensor(loss_cfg.weight, device=self.device, dtype=self.dtype)
            for key, loss_cfg in self.loss_dict.items()
        }

        self.nn = nn
        self.prop = prop
        self.perceptual_models = {
            key: loss_cfg.model
            for key, loss_cfg in self.loss_dict.items()
            if loss_cfg.model is not None
        }

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
            est_dict = {}
            weighted_loss_dict = {}
            full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for key, loss_cfg in self.loss_dict.items():
                loss, estimation = loss_cfg.compute_loss(bin_stim=bin_stim, h_full=h_full)
                weighted_loss = loss * self.loss_weights[key]
                weighted_loss_dict[key] = weighted_loss # For logging purposes only
                full_loss = full_loss + weighted_loss
                if estimation is not None:
                    est_dict[key] = estimation

            # Optimize Losses
            self.optimiser.zero_grad()
            full_loss.backward(retain_graph = True)
            self.optimiser.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Log training in Tensorboard
            if self.logger is not None:
                self.logger.log_losses(weighted_loss_dict, full_loss, epoch)
                self.logger.log_h_full(h_full, epoch)

                if 'localisation' in self.perceptual_models or 'uncertainty' in self.perceptual_models:
                    self.logger.log_bin_cues(bin_stim, epoch)

                if 'localisation' in self.perceptual_models:
                    if self.perceptual_models['localisation'].lstn_pos_conf is not None:
                        self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                   self.prop.spk_pos, 
                                                   est_dict['localisation'], 
                                                   est_dict['uncertainty'] if 'uncertainty' in self.perceptual_models else None, 
                                                   est_dict['coloration'] if 'coloration' in self.perceptual_models else None, 
                                                   epoch, 
                                                   loc_target=self.perceptual_models['localisation'].target[0])
                    else:
                        self.logger.log_plot_arrows(self.prop.lstn_pos, 
                                                   self.prop.spk_pos, 
                                                   est_dict['localisation'], 
                                                   est_dict['uncertainty'] if 'uncertainty' in self.perceptual_models else None, 
                                                   est_dict['coloration'] if 'coloration' in self.perceptual_models else None, 
                                                   epoch)
            
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
                


    def evaluate(self, loss_dict=None):
        if loss_dict is None:
            perceptual_models = self.perceptual_models
        else:
            perceptual_models = {}
            for key, value in loss_dict.items():
                if hasattr(value, "model"):
                    if value.model is not None:
                        perceptual_models[key] = value.model
                else:
                    # Backward compatibility for model-only dicts
                    perceptual_models[key] = value
        # Evaluate final FIRs
        with torch.no_grad():
            h_full = self.nn(torch.ones(1,1, device=self.device, dtype=self.dtype))

            # Propagate sound from speakers
            bin_stim = self.prop(h_full)

            estimation_dict = {}
            # Get perceptual estimations and compute losses
            for entry in perceptual_models.items():
                model = entry[1] # Get model

                model_estimation = model(bin_stim) # Get model estimation
                estimation_dict[entry[0]] = model_estimation.squeeze().detach().cpu().numpy() # Store estimation in dictionary

        return h_full.squeeze(0).detach().cpu().numpy(), estimation_dict, bin_stim.squeeze().detach().cpu().numpy()