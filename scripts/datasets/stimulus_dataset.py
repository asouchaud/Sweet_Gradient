import torch
from torch.utils.data import Dataset, DataLoader

# Duration constants
_STIMULUS_DURATION = 0.01   # seconds
_IMPULSE_DURATION  = 0.01   # seconds (10 ms impulse response)


class StaticStimulusDataset(Dataset):
    """Pre-generated dataset of fixed-length audio stimuli.

    Args:
        fs:             Sampling frequency (Hz).
        num_samples:    Number of stimuli to generate.
        stimulus_type:  One of ``'white_noise'``, ``'impulse'``, or
                        ``'distributed_impulse'``.
        device:         Torch device string (e.g. ``'cpu'``, ``'cuda'``).
        dtype:          Torch dtype for the generated tensors.
    """

    STIMULUS_TYPES = {"white_noise", "impulse", "distributed_impulse"}

    def __init__(
        self,
        fs: int,
        num_samples: int,
        stimulus_type: str = "white_noise",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if stimulus_type not in self.STIMULUS_TYPES:
            raise ValueError(
                f"Unsupported stimulus_type '{stimulus_type}'. "
                f"Choose from: {self.STIMULUS_TYPES}"
            )

        self.fs            = fs
        self.num_samples   = num_samples
        self.stimulus_type = stimulus_type
        self.device        = device
        self.dtype         = dtype
        self.signal_length = int(_STIMULUS_DURATION * fs)

        self.data = self._generate_dataset()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_dataset(self) -> torch.Tensor:
        match self.stimulus_type:
            case "white_noise":
                return self._white_noise()
            case "impulse":
                return self._impulse()
            case "distributed_impulse":
                return self._distributed_impulse()

    def _tensor(self, *shape) -> torch.Tensor:
        """Allocate a zero tensor with the configured device and dtype."""
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)

    def _white_noise(self) -> torch.Tensor:
        return torch.randn(
            self.num_samples, 1, self.signal_length,
            device=self.device, dtype=self.dtype,
        )

    def _impulse(self) -> torch.Tensor:
        # Override length and sample count — all stimuli are identical.
        self.signal_length = int(_IMPULSE_DURATION * self.fs)
        self.num_samples   = 1
        data = self._tensor(self.num_samples, 1, self.signal_length)
        data[:, 0, 0] = 1.0
        return data

    def _distributed_impulse(self) -> torch.Tensor:
        data      = self._tensor(self.num_samples, 1, self.signal_length)
        positions = torch.arange(self.num_samples) % self.signal_length
        data[torch.arange(self.num_samples), 0, positions] = 1.0
        return data


def get_stim_loader(
    fs: int,
    batch_size: int,
    num_samples: int,
    stimulus_type: str = "white_noise",
    shuffle: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[DataLoader, StaticStimulusDataset]:
    """Build a :class:`StaticStimulusDataset` and wrap it in a DataLoader.

    For the ``'impulse'`` stimulus type, ``shuffle`` is forced to ``False``
    and ``batch_size`` to ``1`` (all samples are identical).
    """
    dataset = StaticStimulusDataset(
        fs=fs,
        num_samples=num_samples,
        stimulus_type=stimulus_type,
        device=device,
        dtype=dtype,
    )

    if stimulus_type == "impulse":
        shuffle    = False
        batch_size = 1

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset


# import torch
# from torch.utils.data import Dataset, DataLoader

# class StaticStimulusDataset(Dataset):
#     def __init__(self, fs, num_samples, stimulus_type='white_noise', device = 'cpu', dtype=torch.float64):
#         self.fs = fs
#         self.num_samples = num_samples
#         self.stimulus_type = stimulus_type
#         self.device = device
#         self.dtype = dtype
#         self.signal_length = int(0.05 * fs)

#         self.data = self._generate_dataset()

#     def _generate_dataset(self):
#         if self.stimulus_type == 'white_noise':
#             data = torch.randn(self.num_samples, 1, self.signal_length, device=self.device, dtype=self.dtype)

#         elif self.stimulus_type == 'impulse':
#             self.signal_length = int(0.01*self.fs) # 10 ms impulse response
#             self.num_samples = 1

#             data = torch.zeros(self.num_samples, 1, self.signal_length, device=self.device, dtype=self.dtype)
#             data[:, 0, 0] = 1.0  # Impulse at the beginning of the signal
            
#             print("For stimulus type 'impulse', signal_length is set to 10 ms and num_samples is set to 1 as all of the stimuli are identical impulses")

#         elif self.stimulus_type == 'distributed_impulse':
#             data = torch.zeros(self.num_samples, 1, self.signal_length, device=self.device, dtype=self.dtype)
            
#             # distribute impulses across time
#             positions = torch.arange(self.num_samples) % self.signal_length
#             data[torch.arange(self.num_samples), 0, positions] = 1.0

#         else:
#             raise ValueError(f"Unsupported stimulus_type: {self.stimulus_type}")

#         return data

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.data[idx]
    
# def get_stim_loader(fs, batch_size, num_samples, stimulus_type='white_noise',
#                     shuffle=True, device = 'cpu', dtype=torch.float64):

#     dataset = StaticStimulusDataset(
#         fs=fs,
#         num_samples=num_samples,
#         stimulus_type=stimulus_type,
#         device=device,
#         dtype=dtype
#     )
    
#     if stimulus_type == 'impulse':
#         shuffle = False  # No need to shuffle if all samples are identical impulses
#         batch_size = 1  # Set batch size to 1 for impulse stimulus
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     else:
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#     return loader, dataset
