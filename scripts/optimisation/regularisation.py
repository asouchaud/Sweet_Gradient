import torch



class first_moment_regularisation:
    """
    Regularisation class to compute first moment regularisation loss.
    """
    def __init__(self, n_samples: int, device='cuda', dtype=torch.float32):
        """
        Initialize the regularisation with a weight.
        """
        self.n_samples = n_samples

        self.loss_norm = self.compute_loss(torch.randn(1,1,self.n_samples, device=device, dtype=dtype))

    def compute_loss(self, filter: torch.Tensor) -> torch.Tensor:
        """
        Compute the second moment regularisation loss.

        Args:
            nn_output (torch.Tensor): Output from the neural network.

        Returns:
            torch.Tensor: Computed regularisation loss.
        """

        n_samples = filter.shape[-1]
        device = filter.device
        dtype = filter.dtype
        reg_idx = ((torch.arange(n_samples,0,-1, device=device, dtype=dtype))).view(1,1,-1) # reg_idx is downward because the estimated filters are BACKWARDS IN TIME

        mu = torch.sum((reg_idx*filter.abs()**2), dim = -1)

        return(mu.item())

class second_moment_regularisation:
    """
    Regularisation class to compute second moment regularisation loss.
    """
    def __init__(self, n_samples: int, device='cuda', dtype=torch.float32):
        """
        Initialize the regularisation with a weight.

        """

        self.n_samples = n_samples

        self.loss_norm = self.compute_loss(torch.randn(1,1,self.n_samples, device=device, dtype=dtype))

    def compute_loss(self, filter: torch.Tensor) -> torch.Tensor:
        """
        Compute the second moment regularisation loss.

        Args:
            nn_output (torch.Tensor): Output from the neural network.

        Returns:
            torch.Tensor: Computed regularisation loss.
        """

        n_samples = filter.shape[-1]
        device = filter.device
        dtype = filter.dtype
        reg_idx = ((torch.arange(n_samples,0,-1, device=device, dtype=dtype))).view(1,1,-1) # reg_idx is downward because the estimated filters are BACKWARDS IN TIME

        E = torch.sum((torch.abs(filter)**2), dim = -1)

        mu = torch.sum((reg_idx*filter.abs()**2), dim = -1)/E

        sig = torch.sum((reg_idx - mu[..., None])**2 * filter.abs()**2, dim = -1)/E

        return(sig.item())