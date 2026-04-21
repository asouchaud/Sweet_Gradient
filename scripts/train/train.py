import torch
import numpy as np 
import time

from scripts.plotting.tensorboard_updates import TensorboardLogger

from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation

# Listener-based losses operate on the propagated binaural scene and a perceptual model
# (e.g., localisation, uncertainty, colouration). 
class ListenerBasedLoss:
    def __init__(self, model, weight, device, dtype, listener_weights=None):
        self.model = model
        self.weight = weight
        self.device = device
        self.dtype = dtype

        if listener_weights is not None:
            listener_weights = torch.as_tensor(listener_weights, device=device, dtype=dtype).view(-1)
            if listener_weights.numel() == 0:
                raise ValueError("listener_weights must not be empty.")
            if torch.any(listener_weights < 0):
                raise ValueError("listener_weights must be non-negative.")
            if listener_weights.sum().item() <= 0:
                raise ValueError("listener_weights must sum to a positive value.")
        self.listener_weights = listener_weights

    def _select_listener_weights(self, listener_indices, n_selected_listeners):
        if self.listener_weights is None:
            return None

        if listener_indices is None:
            if self.listener_weights.numel() != n_selected_listeners:
                raise ValueError(
                    "listener_weights and bin_stim disagree on the number of listeners."
                )
            return self.listener_weights

        index = torch.as_tensor(listener_indices, device=self.device, dtype=torch.long)
        return self.listener_weights.index_select(0, index)

    def _compute_weighted_loss(self, model_estimation, target, listener_weights):
        try:
            unreduced_loss = self.model.loss(model_estimation, target, reduction="none")
        except TypeError as exc:
            raise TypeError(
                "listener_weights require a loss function that accepts reduction='none'."
            ) from exc

        if unreduced_loss.ndim == 0:
            raise ValueError("listener_weights require a listener dimension in the loss tensor.")

        listener_dim = 1 if unreduced_loss.ndim > 1 else 0
        if unreduced_loss.shape[listener_dim] != listener_weights.numel():
            raise ValueError(
                "listener_weights and model_estimation disagree on the listener dimension."
            )

        listener_weights = listener_weights.to(device=unreduced_loss.device, dtype=unreduced_loss.dtype)
        weight_sum = listener_weights.sum()

        if weight_sum.item() <= 0:
            return unreduced_loss.sum() * 0.0

        weight_shape = [1] * unreduced_loss.ndim
        weight_shape[listener_dim] = listener_weights.numel()
        weights = listener_weights.view(*weight_shape)

        elements_per_listener = unreduced_loss.numel() / float(unreduced_loss.shape[listener_dim])

        return torch.sum(unreduced_loss * weights) / (weight_sum * elements_per_listener)

    def batch_scale(self, listener_indices, n_listeners):
        if listener_indices is None:
            return 1.0

        if self.listener_weights is None:
            return len(listener_indices) / float(n_listeners)

        if self.listener_weights.numel() != int(n_listeners):
            raise ValueError(
                "listener_weights and the propagation model disagree on the number of listeners."
            )

        index = torch.as_tensor(listener_indices, device=self.device, dtype=torch.long)
        selected_listener_weights = self.listener_weights.index_select(0, index)

        return float((selected_listener_weights.sum() / self.listener_weights.sum()).item())

    def compute_loss(self, bin_stim, spk_filters, stim, listener_indices):  # noqa: ARG002 - kept for uniform interface
        """
        Args:
            bin_stim: Tensor shaped (batch, n_listeners, 2, n_samples), propagated binaural signals, 
            and 2 is the number of ears.
            h_full: Tensor shaped (batch, n_sources, n_samples), source filters (unused here).
            listener_indices: Optional listener subset used for listener mini-batching.

        Why include an unused input:
            Train uses one generic loop over `loss_dict` and always calls
            `compute_loss(bin_stim=..., h_full=..., listener_indices=...)` regardless of loss type.
            Keeping the same method signature across ListenerBasedLoss and
            SourceBasedLoss is intentional polymorphism: it avoids branching
            logic in Train and makes it easy to add new loss classes later.
            For listener-based losses, `h_full` is accepted for API consistency.

        Returns:
            loss: Scalar tensor.
            model_estimation: Tensor output of the perceptual model, shape depends on the model.
        """

        assert hasattr(self.model, "ref_flag")

        if self.model.ref_flag:
            model_estimation = self.model(bin_stim=bin_stim, reference_stim=stim, listener_indices=listener_indices)
        else:
            model_estimation = self.model(bin_stim)

        target = self.model.target[listener_indices]  # [y]
        target = target.unsqueeze(0).expand(model_estimation.shape[0], -1)  # [x, y]


        selected_listener_weights = self._select_listener_weights(
            listener_indices,
            int(bin_stim.shape[1]),
        )

        if selected_listener_weights is None:
            loss = self.model.loss(model_estimation, target) / self.model.loss_norm
        else:
            loss = self._compute_weighted_loss(model_estimation, target, selected_listener_weights) / self.model.loss_norm

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

    def batch_scale(self, listener_indices, n_listeners):
        if listener_indices is None:
            return 1.0
        return len(listener_indices) / float(n_listeners)

    def compute_loss(self, bin_stim, spk_filters, stim):  # noqa: ARG002 - kept for uniform interface
        """
        Args:
            bin_stim: Tensor shaped (batch, n_listeners, 2, n_samples), propagated binaural signals, 
            and 2 is the number of ears (unused here).
            h_full: Tensor shaped (batch, n_sources, n_samples), source filters.
            listener_indices: Optional listener subset used for listener mini-batching (unused here).

        Why include an unused input:
            This class shares the same `compute_loss` signature as
            ListenerBasedLoss so Train can call all losses uniformly.
            Here only `h_full` is needed, while `bin_stim` is accepted for
            API consistency and future extensibility.

        Returns:
            loss: Scalar tensor for the selected source filter.
            estimation: None (source-based losses do not produce perceptual estimates).
        """
        source_filter = spk_filters[:, self.source_idx:self.source_idx + 1, :]
        loss = self.regulariser.compute_loss(source_filter) / self.regulariser.loss_norm
        return loss, None

class Train:
    def __init__(self, n_epoch, stim_loader, lstn_loader, nn, prop, loss_dict, optimiser, loggers=None, scheduler=None, alpha = 1):

        self.n_epoch = n_epoch
        self.stim_loader = stim_loader
        self.lstn_loader = lstn_loader

        self.device = nn[0].weight.device
        self.dtype = nn[0].weight.dtype

        self.optimiser = optimiser

        self.n_samples = nn.n_samples
        self.n_sources = getattr(nn, 'n_sources', 2)
        
        self.loss_dict = loss_dict

        self.nn = nn
        self.prop = prop
        self.perceptual_models = {
            key: loss_cfg.model
            for key, loss_cfg in self.loss_dict.items()
            if loss_cfg.model is not None
        }

        # logger, saver = loggers[0], loggers[1]

        # if logger is not None:
        #     if saver is not None:
        #         self.logger = TensorboardLogger(logger.run_dir / 'tensorboard' / saver.file_name[:-4], device = self.device, dtype = self.dtype)
        #     else: 
        #         self.logger = TensorboardLogger(logger.run_dir / 'tensorboard', device = self.device, dtype = self.dtype)
        # else:
        #     self.logger = None

        # self.scheduler = scheduler
        # self.h_history = []


    def train(self):
        unit_input = torch.ones(1,1, device=self.device, dtype=self.dtype)

        for epoch in range(self.n_epoch):
            full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for stimulus in self.stim_loader:
                for listener_idx, listener_pos in self.lstn_loader:
                    
                    est_dict = {}
                    batch_full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

                    spk_filters = self.nn(unit_input)
                    bin_stim = self.prop(stimulus, spk_filters, listener_idx)

                    for key, loss_cfg in self.loss_dict.items():

                        loss, estimation = loss_cfg.compute_loss(bin_stim=bin_stim, spk_filters=spk_filters, stim=stimulus, listener_indices=listener_idx)
                        weigthed_loss = loss * loss_cfg.weight
                        
                        batch_full_loss += weigthed_loss
                        est_dict[key] = estimation

                    self.optimiser.zero_grad()
                    batch_full_loss.backward()
                    full_loss = full_loss + batch_full_loss.detach()

                    self.optimiser.step()
                    #     time_epoch = time.time() - time_epoch_start
            
            # Print progress
            print(
                f"\rEpoch: {epoch+1:<4d} ¦¦ "
                f"Loss: {full_loss.item():<10.5f} ¦¦ ",
                # f"LR: {current_lr:.6f} ¦¦ "
                # f"Time: {time_epoch:.3f} seconds¦¦",
                end="",
                flush=True
            )
            print()


    def evaluate(self, eval_lstn_pos, eval_stimulus_data_loader, loss_dict=None):

        self.prop = BinauralFreeFieldPropagation(
            fs=self.prop.fs,
            lstn_pos=eval_lstn_pos,
            spk_pos=self.prop.spk_pos,
            sofa_file_path=self.prop.sources[0].sofa_file_path,
            apply_distance_attenuation=self.prop.apply_distance_attenuation,
            device=self.device,
            dtype=self.dtype
        )

        active_loss_dict = loss_dict if loss_dict is not None else self.loss_dict

        estimation_dict = {key: [] for key in active_loss_dict}
        bin_stim_list = []

        # Use all listener indices at once
        listener_idx = torch.arange(len(eval_lstn_pos), device=self.device)

        with torch.no_grad():
            for stimulus in eval_stimulus_data_loader:

                spk_filters = self.nn(torch.ones(1, 1, device=self.device, dtype=self.dtype))
                bin_stim = self.prop(stimulus, spk_filters, listener_idx)
                bin_stim_list.append(bin_stim.squeeze().detach().cpu().numpy())
                for key, loss_cfg in active_loss_dict.items():
                    _, estimation = loss_cfg.compute_loss(
                        bin_stim=bin_stim,
                        spk_filters=spk_filters,
                        stim=stimulus,
                        listener_indices=listener_idx
                    )
                    estimation_dict[key] = estimation.squeeze().detach().cpu().numpy()

        # Average across all stimuli
        estimation_dict = {key: np.mean(np.stack(vals, axis=0), axis=0) for key, vals in estimation_dict.items()}
        bin_stim_out = np.mean(np.stack(bin_stim_list, axis=0), axis=0)

        return spk_filters.squeeze(0).detach().cpu().numpy(), estimation_dict, bin_stim_out

    # def evaluate(self, eval_lstn_pos, eval_stimulus_data_loader, loss_dict=None):
  
    #     self.prop = BinauralFreeFieldPropagation(fs=self.prop.fs, lstn_pos=eval_lstn_pos, spk_pos=self.prop.spk_pos, sofa_file_path=self.prop.sofa_file_path, apply_distance_attenuation=self.prop.apply_distance_attenuation, device=self.device, dtype=self.dtype)

    #     if loss_dict is None:
    #         perceptual_models = self.perceptual_models
    #     else:
    #         perceptual_models = {}
    #         for key, value in loss_dict.items():
    #             if hasattr(value, "model"):
    #                 if value.model is not None:
    #                     perceptual_models[key] = value.model
    #             else:
    #                 # Backward compatibility for model-only dicts
    #                 perceptual_models[key] = value
    #     # Evaluate final FIRs
    #     with torch.no_grad():
    #         for eval_stimulus in eval_stimulus_data_loader:
    #             spk_filters = self.nn(torch.ones(1,1, device=self.device, dtype=self.dtype))

    #             # Propagate sound from speakers
    #             bin_stim = self.prop(eval_stimulus, spk_filters)

    #             estimation_dict = {}
    #             # Get perceptual estimations and compute losses
    #             for entry in perceptual_models.items():
    #                 model = entry[1] # Get model

    #                 model_estimation = model(bin_stim) # Get model estimation
    #                 estimation_dict[entry[0]] = model_estimation.squeeze().detach().cpu().numpy() # Store estimation in dictionary

    #         return spk_filters.squeeze(0).detach().cpu().numpy(), estimation_dict, bin_stim.squeeze().detach().cpu().numpy()
                    
                
    def _log_tensorboard(self, full_loss, weighted_loss, estimation_dict, bin_stim, spk_filters, listener_pos, epoch):
        if self.logger is not None:
            self.logger.log_losses(weighted_loss, full_loss, epoch)
            self.logger.log_h_full(spk_filters, epoch)

            if 'localisation' in self.perceptual_models or 'uncertainty' in self.perceptual_models:
                self.logger.log_bin_cues(bin_stim, epoch)

            if 'localisation' in self.perceptual_models:
                self.logger.log_plot_arrows(listener_pos,
                                                self.prop.spk_pos, 
                                                estimation_dict['localisation'], 
                                                estimation_dict['uncertainty'] if 'uncertainty' in self.perceptual_models else None, 
                                                estimation_dict['coloration'] if 'coloration' in self.perceptual_models else None, 
                                                epoch, 
                                                target_pos=getattr(self.perceptual_models['localisation'], 'target_pos', None))
        

