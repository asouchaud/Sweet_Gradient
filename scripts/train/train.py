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

    def compute_loss(self, bin_stim, h_full, listener_indices=None):  # noqa: ARG002 - kept for uniform interface
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
        if getattr(self.model, "supports_listener_indices", False):
            model_estimation = self.model(bin_stim, listener_indices=listener_indices)
        else:
            model_estimation = self.model(bin_stim)

        target = torch.as_tensor(self.model.target, device=self.device, dtype=self.dtype)
        if listener_indices is not None and target.numel() > 1:
            target = target[listener_indices]
        target = target.expand_as(model_estimation)

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

    def compute_loss(self, bin_stim, h_full, listener_indices=None):  # noqa: ARG002 - kept for uniform interface
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
        source_filter = h_full[:, self.source_idx:self.source_idx + 1, :]
        loss = self.regulariser.compute_loss(source_filter) / self.regulariser.loss_norm
        return loss, None

class Train:
    def __init__(self, n_epoch, nn, prop, loss_dict, optimiser, loggers=None, scheduler=None, alpha = 1, listener_batch_size=None, symmetric_listener_sampling=False):

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

        self.listener_batch_size = None if listener_batch_size is None else int(listener_batch_size)
        self.symmetric_listener_sampling = symmetric_listener_sampling
        self.listener_groups = None

        if self.listener_batch_size is not None:
            if self.listener_batch_size <= 0:
                raise ValueError("listener_batch_size must be a positive integer.")
            if not hasattr(self.prop, 'n_listeners') or not hasattr(self.prop, 'lstn_pos'):
                raise ValueError("listener_batch_size requires a propagation model exposing n_listeners and lstn_pos.")
            self.listener_batch_size = min(self.listener_batch_size, int(self.prop.n_listeners))
            if self.symmetric_listener_sampling and self.listener_batch_size < int(self.prop.n_listeners):
                self.listener_groups = self._build_symmetric_listener_groups()

    def _build_symmetric_listener_groups(self):
        listener_groups = {}
        lstn_pos = self.prop.lstn_pos.detach().cpu().tolist()

        for idx, (x, y) in enumerate(lstn_pos):
            key = (round(float(x), 8), round(abs(float(y)), 8))
            if key not in listener_groups:
                listener_groups[key] = []
            listener_groups[key].append(idx)

        grouped_indices = [sorted(group) for group in listener_groups.values()]
        invalid_groups = [group for group in grouped_indices if len(group) not in (1, 2)]
        if invalid_groups:
            raise ValueError(
                "Symmetric listener sampling expects mirrored pairs and optional self-symmetric listeners."
            )

        return sorted(grouped_indices, key=lambda group: group[0])

    def _sample_listener_indices(self):
        if self.listener_batch_size is None or self.listener_batch_size >= int(self.prop.n_listeners):
            return None

        if not self.symmetric_listener_sampling:
            return sorted(torch.randperm(int(self.prop.n_listeners)).tolist()[:self.listener_batch_size])

        pair_groups = [group for group in self.listener_groups if len(group) == 2]
        singleton_groups = [group for group in self.listener_groups if len(group) == 1]
        selected_listener_indices = []

        if self.listener_batch_size % 2 == 1:
            if len(singleton_groups) == 0:
                raise ValueError(
                    "Odd listener_batch_size with symmetric sampling requires at least one self-symmetric listener."
                )
            singleton_order = torch.randperm(len(singleton_groups)).tolist()
            selected_listener_indices.extend(singleton_groups[singleton_order[0]])

        n_pairs = (self.listener_batch_size - len(selected_listener_indices)) // 2
        if n_pairs > len(pair_groups):
            raise ValueError("listener_batch_size is too large for symmetric listener sampling.")

        pair_order = torch.randperm(len(pair_groups)).tolist()
        for pair_idx in pair_order[:n_pairs]:
            selected_listener_indices.extend(pair_groups[pair_idx])

        return sorted(selected_listener_indices)

    def _build_listener_batches(self):
        if self.listener_batch_size is None or self.listener_batch_size >= int(self.prop.n_listeners):
            return [None]

        if not self.symmetric_listener_sampling:
            shuffled_indices = torch.randperm(int(self.prop.n_listeners)).tolist()
            return [
                sorted(shuffled_indices[start:start + self.listener_batch_size])
                for start in range(0, len(shuffled_indices), self.listener_batch_size)
            ]

        shuffled_groups = [
            self.listener_groups[group_idx]
            for group_idx in torch.randperm(len(self.listener_groups)).tolist()
        ]

        listener_batches = []
        current_batch = []
        current_batch_size = 0

        for group in shuffled_groups:
            group_size = len(group)
            if group_size > self.listener_batch_size:
                raise ValueError("listener_batch_size is too small for symmetric listener sampling.")

            if current_batch_size + group_size > self.listener_batch_size and current_batch:
                listener_batches.append(sorted(current_batch))
                current_batch = list(group)
                current_batch_size = group_size
            else:
                current_batch.extend(group)
                current_batch_size += group_size

        if current_batch:
            listener_batches.append(sorted(current_batch))

        return listener_batches

    def train(self):
        unit_input = torch.ones(1,1, device=self.device, dtype=self.dtype)

        for epoch in range(self.n_epoch):
            time_epoch_start = time.time()

            listener_batches = self._build_listener_batches()

            est_dict = {}
            weighted_loss_dict = {}
            full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            bin_stim = None
            listener_indices = None
            h_full_snapshot = None

            for batch_idx, listener_indices in enumerate(listener_batches):
                # Estimate FIRs
                h_full = self.nn(unit_input)

                if batch_idx == 0:
                    h_full_snapshot = h_full.detach()
                    self.h_history.append(h_full_snapshot.squeeze(0).cpu().numpy())

                # Propagate sound from speakers
                if listener_indices is None:
                    bin_stim = self.prop(h_full)
                else:
                    bin_stim = self.prop(h_full, listener_indices=listener_indices)

                batch_scale = 1.0 if listener_indices is None else len(listener_indices) / float(self.prop.n_listeners)

                # Get perceptual estimations and compute losses
                est_dict = {}
                batch_full_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for key, loss_cfg in self.loss_dict.items():
                    loss, estimation = loss_cfg.compute_loss(bin_stim=bin_stim, h_full=h_full, listener_indices=listener_indices)
                    weighted_loss = loss * self.loss_weights[key] * batch_scale
                    if key in weighted_loss_dict:
                        weighted_loss_dict[key] = weighted_loss_dict[key] + weighted_loss.detach()
                    else:
                        weighted_loss_dict[key] = weighted_loss.detach()
                    batch_full_loss = batch_full_loss + weighted_loss
                    if estimation is not None:
                        est_dict[key] = estimation

                self.optimiser.zero_grad()
                batch_full_loss.backward()
                full_loss = full_loss + batch_full_loss.detach()

                self.optimiser.step()

            if self.scheduler is not None: # Update learning rate once per epoch
                self.scheduler.step()

            # Log training in Tensorboard
            if self.logger is not None:
                self.logger.log_losses(weighted_loss_dict, full_loss, epoch)
                self.logger.log_h_full(h_full_snapshot, epoch)

                if 'localisation' in self.perceptual_models or 'uncertainty' in self.perceptual_models:
                    self.logger.log_bin_cues(bin_stim, epoch)

                if 'localisation' in self.perceptual_models:
                    current_lstn_pos = (
                        self.prop.lstn_pos
                        if listener_indices is None
                        else self.prop.lstn_pos[listener_indices]
                    )
                    self.logger.log_plot_arrows(current_lstn_pos,
                                                   self.prop.spk_pos, 
                                                   est_dict['localisation'], 
                                                   est_dict['uncertainty'] if 'uncertainty' in self.perceptual_models else None, 
                                                   est_dict['coloration'] if 'coloration' in self.perceptual_models else None, 
                                                   epoch, 
                                                   target_pos=getattr(self.perceptual_models['localisation'], 'target_pos', None))
            
            current_lr = self.optimiser.param_groups[0]['lr']
            time_epoch = time.time() - time_epoch_start
            # Print progress
            print(
                f"\rEpoch: {epoch+1:<4d} ¦¦ "
                f"Loss: {full_loss.item():<10.5f} ¦¦ ",
                f"LR: {current_lr:.6f} ¦¦ "
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