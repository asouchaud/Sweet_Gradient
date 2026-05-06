import torch

from scripts.datasets.listener_dataset import SymmetricListenerBatchSampler


class JointStimulusListenerBatchLoader:
    """
    Iterable joint loader that yields rectangular stimulus x listener mini-batches:
        stimulus:      (stim_batch, 1, T)
        listener_idx:  (listener_batch,)
        listener_pos:  (listener_batch, 2)
    """

    def __init__(self, stimulus_dataset, lstn_pos, stim_batch_size, listener_batch_size, symmetric_listener_sampling=True, shuffle_stimulus=True):
        self.stimulus_dataset = stimulus_dataset
        self.lstn_pos = lstn_pos
        self.stim_batch_size = int(stim_batch_size)
        self.listener_batch_size = int(listener_batch_size)
        self.symmetric_listener_sampling = bool(symmetric_listener_sampling)
        self.shuffle_stimulus = bool(shuffle_stimulus)

        if self.stim_batch_size <= 0:
            raise ValueError("stim_batch_size must be a positive integer.")
        if self.listener_batch_size <= 0:
            raise ValueError("listener_batch_size must be a positive integer.")

        self._precomputed_batches = []
        self.rebuild_batches()

    def _stimulus_batch_indices(self):
        n_stim = self.stimulus_dataset.data.shape[0]
        if self.shuffle_stimulus:
            stim_indices = torch.randperm(n_stim, device=self.stimulus_dataset.data.device)
        else:
            stim_indices = torch.arange(n_stim, device=self.stimulus_dataset.data.device)

        for i in range(0, n_stim, self.stim_batch_size):
            yield stim_indices[i:i + self.stim_batch_size]

    def _listener_batches(self):
        sampler = SymmetricListenerBatchSampler(
            self.lstn_pos,
            batch_size=self.listener_batch_size,
            symmetric=self.symmetric_listener_sampling
        )
        for batch in sampler:
            yield torch.as_tensor(batch, device=self.lstn_pos.device, dtype=torch.long)

    def rebuild_batches(self):
        self._precomputed_batches = []
        listener_batches = list(self._listener_batches())
        for stim_idx in self._stimulus_batch_indices():
            stimulus = self.stimulus_dataset.data.index_select(0, stim_idx)
            for listener_idx in listener_batches:
                listener_pos = self.lstn_pos.index_select(0, listener_idx)
                self._precomputed_batches.append((stimulus, listener_idx, listener_pos))

    def __iter__(self):
        for batch in self._precomputed_batches:
            yield batch

    def __len__(self):
        return len(self._precomputed_batches)

