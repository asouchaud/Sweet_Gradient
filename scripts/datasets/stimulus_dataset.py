
import torch
from torch.utils.data import Dataset, DataLoader

class StaticStimulusDataset(Dataset):
    def __init__(self, fs, num_samples, stimulus_type='white_noise', device = 'cpu', dtype=torch.float64):
        self.fs = fs
        self.num_samples = num_samples
        self.stimulus_type = stimulus_type
        self.device = device
        self.dtype = dtype
        self.signal_length = int(0.01 * fs)

        self.data = self._generate_dataset()

    def _generate_dataset(self):
        if self.stimulus_type == 'white_noise':
            data = torch.randn(self.num_samples, 1, self.signal_length, dtype=self.dtype)

        elif self.stimulus_type == 'impulse':
            data = torch.zeros(self.num_samples, 1, self.signal_length, dtype=self.dtype)
            
            # distribute impulses across time
            positions = torch.arange(self.num_samples) % self.signal_length
            data[torch.arange(self.num_samples), 0, positions] = 1.0

        else:
            raise ValueError(f"Unsupported stimulus_type: {self.stimulus_type}")

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
    
def get_stim_loader(fs, batch_size, num_samples, stimulus_type='white_noise',
                    shuffle=True, device = 'cpu', dtype=torch.float64):

    dataset = StaticStimulusDataset(
        fs=fs,
        num_samples=num_samples,
        stimulus_type=stimulus_type,
        dtype=dtype
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, dataset
