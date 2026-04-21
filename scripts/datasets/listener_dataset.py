import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# Listener Dataset Class 
class ListenerDataset(torch.utils.data.Dataset):
    def __init__(self, lstn_pos):
        assert isinstance(lstn_pos, torch.Tensor), "lstn_pos must be a torch.Tensor"
        assert lstn_pos.ndim == 2, "lstn_pos must have shape (n_listeners, coordinates)"
        assert lstn_pos.shape[1] == 2, "Coordinates must be 2D (x, y)"
        self.lstn_pos = lstn_pos  
        self.device = lstn_pos.device
        self.dtype = lstn_pos.dtype

    def __len__(self):
        return self.lstn_pos.shape[0]

    def __getitem__(self, idx):
        return idx, self.lstn_pos[idx]  
    
# Sampler which can sample listeners symmetrically compared to the sweet spot
class SymmetricListenerBatchSampler(Sampler):
    def __init__(self, lstn_pos, batch_size, symmetric=True):
        
        self.lstn_pos = lstn_pos
        self.symmetric = symmetric
        self.n_listeners = lstn_pos.shape[0]

        self.batch_size = batch_size
        if batch_size > len(lstn_pos):
            print("Listener batch size greater than number of listener. Setting batch size to number of listeners")
            self.batch_size = len(lstn_pos)

        if symmetric:
            self.listener_groups = self._build_groups()
        else:
            self.listener_groups = None

    def _build_groups(self):
        listener_groups = {}
        lstn_pos = self.lstn_pos.detach().cpu().tolist()

        for idx, (x, y) in enumerate(lstn_pos):
            key = (round(float(x), 8), round(abs(float(y)), 8))
            listener_groups.setdefault(key, []).append(idx)

        grouped = [sorted(g) for g in listener_groups.values()]

        # validation
        for g in grouped:
            if len(g) not in (1, 2):
                raise ValueError("Invalid symmetric grouping")

        return grouped

    def __iter__(self):
        if not self.symmetric:
            indices = torch.randperm(self.n_listeners).tolist()
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
            return

        # symmetric batching
        shuffled_groups = [
            self.listener_groups[i]
            for i in torch.randperm(len(self.listener_groups))
        ]

        batch = []
        size = 0

        for group in shuffled_groups:
            gsize = len(group)

            if size + gsize > self.batch_size and batch:
                yield sorted(batch)
                batch = list(group)
                size = gsize
            else:
                batch.extend(group)
                size += gsize

        if batch:
            yield sorted(batch)

    def __len__(self):
        return self.n_listeners // self.batch_size
    

# Function to create listener dataloader 
def get_listener_loader(lstn_pos, batch_size, symmetric=True):
    dataset = ListenerDataset(lstn_pos)

    sampler = SymmetricListenerBatchSampler(
        lstn_pos,
        batch_size=batch_size,
        symmetric=symmetric
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler
    )

    return loader, dataset