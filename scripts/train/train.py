import torch
import numpy as np 
import time

from scripts.plotting.tensorboard_updates import TensorboardLogger
from scripts.helpers.localisation_tracking import (
    init_localisation_cue_tracking,
    update_localisation_cue_tracking,
    finalize_localisation_cue_tracking,
)

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
            spk_filters: Tensor shaped (batch, n_sources, n_samples), source filters (unused here).
            listener_indices: Optional listener subset used for listener mini-batching.

        Why include an unused input:
            Train uses one generic loop over `loss_dict` and always calls
            `compute_loss(bin_stim=..., spk_filters=..., listener_indices=...)` regardless of loss type.
            Keeping the same method signature across ListenerBasedLoss and
            SourceBasedLoss is intentional polymorphism: it avoids branching
            logic in Train and makes it easy to add new loss classes later.
            For listener-based losses, `spk_filters` is accepted for API consistency.

        Returns:
            loss: Scalar tensor.
            model_estimation: Tensor output of the perceptual model, shape depends on the model.
        """

        assert hasattr(self.model, "ref_flag")

        if self.model.ref_flag:
            model_estimation = self.model(bin_stim=bin_stim, reference_stim=stim, listener_indices=listener_indices)
        else:
            model_estimation = self.model(bin_stim)

        target_indices = torch.as_tensor(listener_indices, device=self.model.target.device, dtype=torch.long)
        target = self.model.target[target_indices]  # [y]
        target = target.unsqueeze(0).expand(model_estimation.shape[0], -1)  # [x, y]
        target = target.to(device=model_estimation.device, dtype=model_estimation.dtype)


        selected_listener_weights = self._select_listener_weights(
            listener_indices,
            int(bin_stim.shape[1]),
        )

        if selected_listener_weights is None:
            loss = self.model.loss(model_estimation, target) / self.model.loss_norm
        else:
            loss = self._compute_weighted_loss(model_estimation, target, selected_listener_weights) / self.model.loss_norm

        return loss, model_estimation


# Source-based losses regularise individual source filters directly from spk_filters.
# They do not expose a perceptual model, so Train treats them as optimization-only
# objectives (no perceptual estimation output needed for evaluate/plotting).
class SourceBasedLoss:
    def __init__(self, regulariser, source_idx, weight):
        self.model = None
        self.regulariser = regulariser
        if isinstance(source_idx, int):
            source_indices = [source_idx]
        elif torch.is_tensor(source_idx):
            source_indices = source_idx.detach().cpu().tolist()
        else:
            source_indices = list(source_idx)

        if len(source_indices) == 0:
            raise ValueError("source_idx must contain at least one source index.")

        self.source_indices = tuple(int(idx) for idx in source_indices)
        self.weight = weight

    def batch_scale(self, listener_indices, n_listeners):
        if listener_indices is None:
            return 1.0
        return len(listener_indices) / float(n_listeners)

    def compute_loss(self, bin_stim, spk_filters, stim, listener_indices=None):  # noqa: ARG002 - kept for uniform interface
        """
        Args:
            bin_stim: Tensor shaped (batch, n_listeners, 2, n_samples), propagated binaural signals, 
            and 2 is the number of ears (unused here).
            spk_filters: Tensor shaped (batch, n_sources, n_samples), source filters.
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
        source_indices = torch.as_tensor(
            self.source_indices,
            device=spk_filters.device,
            dtype=torch.long,
        ).view(-1)
        source_filter = spk_filters.index_select(1, source_indices)
        loss = self.regulariser.compute_loss(source_filter) / self.regulariser.loss_norm
        return loss, None

class Train:
    def __init__(
        self,
        n_epoch,
        stim_loader,
        lstn_loader,
        nn,
        prop,
        loss_dict,
        optimiser,
        logger=None,
        saver=None,
        loggers=None,
        scheduler=None,
        alpha=1,
    ):

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
        self.localisation_model = self.perceptual_models.get("localisation", None)
        self.localisation_tracking = init_localisation_cue_tracking(self.localisation_model)
        self.localisation_grad_tracking = {
            "epoch": [],
            "itd_grad_norm": [],
            "ild_grad_norm": [],
            "itd_grad_band_norm": [],
            "ild_grad_band_norm": [],
            "itd_grad_band_share": [],
            "ild_grad_band_share": [],
        }


        self.logger = self._build_tensorboard_logger(logger=logger, saver=saver, loggers=loggers)

        # self.scheduler = scheduler
        # self.h_history = []


    def train(self):
        unit_input = torch.ones(1,1, device=self.device, dtype=self.dtype)

        for epoch in range(self.n_epoch):
            full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            ## For storing per-listener and per-loss estimations across batches for logging and plotting at the end of the epoch
            accum_est = {
                (lstn_idx, key): []
                for lstn_idx in range(len(self.lstn_loader.dataset))
                for key in self.loss_dict
                if isinstance(self.loss_dict[key], ListenerBasedLoss)
            }

            # For storing per-loss values across batches for logging and plotting at the end of the epoch
            accum_loss = {
                    key: []
                    for key in self.loss_dict
            }
            accum_localisation_grad_norm = {"itd": [], "ild": [], "itd_band": [], "ild_band": []}

            # For storing binaural stimulus for each listener across batches for logging and plotting at the end of the epoch
            accum_bin_stim = {
                lstn_idx: []
                for lstn_idx in range(len(self.lstn_loader.dataset))
            }
            ##################################################################

            for stimulus in self.stim_loader:
                for listener_idx, listener_pos in self.lstn_loader:
                    # Initialize batch loss for listener mini-batch
                    batch_full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

                    # Estimate loudspeaker filters and propagate stimulus for the current listener mini-batch 
                    # to get binaural signals at the ear for the current stimulus and listener subset
                    spk_filters = self.nn(unit_input)
                    bin_stim = self.prop(stimulus, spk_filters, listener_idx)
                    
                    # Store binaural stimulus for each listener for logging and plotting at the end of the epoch
                    for i, idx in enumerate(listener_idx):
                        accum_bin_stim[idx.item()].append(bin_stim[:,i,:,:].detach().cpu()) # 
                    
                    # Compute and accumulate weighted loss across all perceptual models for the current stimulus and listener subset
                    for key, loss_cfg in self.loss_dict.items():
                        # Compute loss and model estimation for the current batch and listener subset
                        loss, estimation = loss_cfg.compute_loss(bin_stim=bin_stim, spk_filters=spk_filters, stim=stimulus, listener_indices=listener_idx)
                        weighted_loss = loss * loss_cfg.weight
                        
                        # Accumulate weighted loss for backpropagation
                        batch_full_loss += weighted_loss
                        
                        ## Store estimations for logging and plotting at the end of the epoch
                        for i, idx in enumerate(listener_idx):
                            accum_est[(idx.item(), key)].append(estimation[:,i].detach().cpu()) if isinstance(loss_cfg, ListenerBasedLoss) else None
                        accum_loss[key].append(weighted_loss.detach().cpu().item())

                    # Backpropagate total loss for the current batch and listener subset
                    self.optimiser.zero_grad()
                    batch_full_loss.backward()
                    if self.localisation_model is not None and hasattr(self.localisation_model, "latest_interauralcues"):
                        itd_grad = self.localisation_model.latest_interauralcues["itd"].grad
                        ild_grad = self.localisation_model.latest_interauralcues["ild"].grad
                        if itd_grad is not None:
                            accum_localisation_grad_norm["itd"].append(torch.norm(itd_grad).detach().cpu().item())
                            accum_localisation_grad_norm["itd_band"].append(
                                torch.linalg.vector_norm(itd_grad, dim=(0, 1)).detach().cpu().numpy()
                            )
                        if ild_grad is not None:
                            accum_localisation_grad_norm["ild"].append(torch.norm(ild_grad).detach().cpu().item())
                            accum_localisation_grad_norm["ild_band"].append(
                                torch.linalg.vector_norm(ild_grad, dim=(0, 1)).detach().cpu().numpy()
                            )
                    full_loss = full_loss + batch_full_loss.detach()

                    # Update filters
                    self.optimiser.step()
                    #     time_epoch = time.time() - time_epoch_start
            
            self._print_epoch_progress(epoch, full_loss)

            ## Average estimations across stimulus batches for each listener and loss type for logging and plotting
            avg_est = {
                (lstn_idx, key): torch.stack(vals).mean()
                for (lstn_idx, key), vals in accum_est.items()
            }

            # Average losses across stimulus batches for each loss type for logging and plotting
            avg_loss = {
                key: np.mean(vals)
                for key, vals in accum_loss.items()
            }

            # Estimate filters after last update for logging
            spk_filters = self.nn(unit_input)
            mean_itd_grad_norm = float(np.mean(accum_localisation_grad_norm["itd"])) if len(accum_localisation_grad_norm["itd"]) > 0 else np.nan
            mean_ild_grad_norm = float(np.mean(accum_localisation_grad_norm["ild"])) if len(accum_localisation_grad_norm["ild"]) > 0 else np.nan
            mean_itd_grad_band_norm = (
                np.mean(np.stack(accum_localisation_grad_norm["itd_band"], axis=0), axis=0)
                if len(accum_localisation_grad_norm["itd_band"]) > 0
                else np.array([])
            )
            mean_ild_grad_band_norm = (
                np.mean(np.stack(accum_localisation_grad_norm["ild_band"], axis=0), axis=0)
                if len(accum_localisation_grad_norm["ild_band"]) > 0
                else np.array([])
            )
            mean_itd_grad_band_share = mean_itd_grad_band_norm / (np.sum(mean_itd_grad_band_norm) + 1e-12) if mean_itd_grad_band_norm.size > 0 else np.array([])
            mean_ild_grad_band_share = mean_ild_grad_band_norm / (np.sum(mean_ild_grad_band_norm) + 1e-12) if mean_ild_grad_band_norm.size > 0 else np.array([])
            self.localisation_grad_tracking["epoch"].append(epoch)
            self.localisation_grad_tracking["itd_grad_norm"].append(mean_itd_grad_norm)
            self.localisation_grad_tracking["ild_grad_norm"].append(mean_ild_grad_norm)
            self.localisation_grad_tracking["itd_grad_band_norm"].append(mean_itd_grad_band_norm.tolist())
            self.localisation_grad_tracking["ild_grad_band_norm"].append(mean_ild_grad_band_norm.tolist())
            self.localisation_grad_tracking["itd_grad_band_share"].append(mean_itd_grad_band_share.tolist())
            self.localisation_grad_tracking["ild_grad_band_share"].append(mean_ild_grad_band_share.tolist())

            if self.localisation_model is not None:
                update_localisation_cue_tracking(
                    tracking=self.localisation_tracking,
                    epoch=epoch,
                    accum_bin_stim=accum_bin_stim,
                    fs=self.localisation_model.fs,
                    gtfb=self.localisation_model.gtfb,
                    tau_iacc=self.localisation_model.tau_IACC,
                    localisation_model=self.localisation_model,
                )

            self._log_tensorboard(
                full_loss=full_loss,
                weighted_loss=avg_loss,
                estimation_dict=avg_est,
                bin_stim=accum_bin_stim,
                spk_filters=spk_filters,
                listener_pos=self.lstn_loader.dataset,
                epoch=epoch
            )

        self.localisation_tracking = finalize_localisation_cue_tracking(self.localisation_tracking)

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
                bin_stim_list.append(bin_stim.detach().cpu().numpy())

                for key, loss_cfg in active_loss_dict.items():
                    _, estimation = loss_cfg.compute_loss(
                        bin_stim=bin_stim,
                        spk_filters=spk_filters,
                        stim=stimulus,
                        listener_indices=listener_idx
                    )
                    # Source-based losses may return no perceptual estimation.
                    if estimation is None:
                        continue
                    # estimation shape: n_stimulus_in_batch x n_listeners
                    estimation_dict[key].append(estimation.detach().cpu().numpy())  # ← append, not overwrite

        # Concatenate across batches along stimulus axis (axis=0), then average over stimulus
        estimation_dict = {
            key: (
                np.mean(np.concatenate(vals, axis=0), axis=0)  # ← concatenate then average over stimulus
                if len(vals) > 0
                else None
            )
            for key, vals in estimation_dict.items()
        }
        bin_stim_out = np.mean(np.concatenate(bin_stim_list, axis=0), axis=0)

        return spk_filters.squeeze(0).detach().cpu().numpy(), estimation_dict, bin_stim_out


        
    def _log_tensorboard(self, full_loss, weighted_loss, estimation_dict, bin_stim, spk_filters, listener_pos, epoch):
        if self.logger is not None:
            self.logger.log_losses(weighted_loss, full_loss, epoch)
            self.logger.log_spk_filters(spk_filters, epoch)

            has_binaural_cue_model = any(
                loss_name in {"localisation", "uncertainty"}
                for (_, loss_name) in estimation_dict.keys()
            )
            if has_binaural_cue_model:
                self.logger.log_bin_cues(bin_stim, epoch)

            self._log_arrow_plot(estimation_dict, epoch)

    def _build_tensorboard_logger(self, logger=None, saver=None, loggers=None):
        run_logger = logger

        # Backwards-compatible fallback: legacy positional logger container.
        if run_logger is None and loggers is not None:
            if isinstance(loggers, (list, tuple)):
                if len(loggers) > 0:
                    run_logger = loggers[0]
                if saver is None and len(loggers) > 1:
                    saver = loggers[1]
            else:
                run_logger = loggers

        if run_logger is None or not hasattr(run_logger, "run_dir"):
            return None

        log_dir = run_logger.run_dir / "tensorboard"
        if saver is not None and hasattr(saver, "file_name"):
            log_dir = log_dir / saver.file_name[:-4]

        return TensorboardLogger(log_dir, device=self.device, dtype=self.dtype)

    def _print_epoch_progress(self, epoch, full_loss):
        print(
            f"\rEpoch: {epoch+1:<4d} ¦¦ "
            f"Loss: {full_loss.item():<10.5f} ¦¦ ",
            end="",
            flush=True,
        )
        print()

    def _extract_listener_metric(self, estimation_dict, metric_name):
        n_listeners = len(self.lstn_loader.dataset)
        values = []
        for lstn_idx in range(n_listeners):
            value = estimation_dict.get((lstn_idx, metric_name), None)
            if value is None:
                values.append(np.nan)
            elif torch.is_tensor(value):
                values.append(float(value.detach().cpu().item()))
            else:
                values.append(float(value))
        return torch.tensor(values, device=self.device, dtype=self.dtype)

    def _log_arrow_plot(self, estimation_dict, epoch):
        loss_names = {loss_name for (_, loss_name) in estimation_dict.keys()}
        if "localisation" not in loss_names:
            return

        loc = self._extract_listener_metric(estimation_dict, "localisation")
        unc = self._extract_listener_metric(estimation_dict, "uncertainty") if "uncertainty" in loss_names else None
        col = self._extract_listener_metric(estimation_dict, "colouration") if "colouration" in loss_names else None

        self.logger.log_plot_arrows(
            lstn_pos=self.prop.lstn_pos,
            spk_pos=self.prop.spk_pos,
            loc=loc,
            unc=unc,
            col=col,
            epoch=epoch,
            target_pos=getattr(self.perceptual_models.get("localisation", None), "target_pos", None),
        )
        

