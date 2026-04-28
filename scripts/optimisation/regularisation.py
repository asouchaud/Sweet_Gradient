import torch

    
def rms_normalisation(h, eps = 1e-8):
    
    rms = torch.sqrt(torch.mean(h**2, dim=-1, keepdim = True))

    h_norm = h/(rms + eps)
    
    return(h_norm)

def energy_normalisation(h, eps = 1e-8):
    
    E = torch.sum(h**2, dim=-1, keepdim = True)

    h_norm = h/(E + eps)

    return(h_norm)

class first_moment_regularisation:
    """
    Regularisation class to compute first moment regularisation loss.
    """
    def __init__(self, n_samples: int, device='cuda', dtype=torch.float32, eps = 1e-8):
        """
        Initialize the regularisation with a weight.
        """

        self.multi_flag = False

        self.eps = eps
        self.n_samples = n_samples
        self.reg_idx = ((torch.arange(n_samples, device=device, dtype=dtype))).view(1,1,-1)

        self.loss_norm = self.compute_loss(torch.randn(1,1,self.n_samples, device=device, dtype=dtype))
        # self.loss_norm = 1

    def compute_loss(self, filter: torch.Tensor) -> torch.Tensor:
        """
        Compute the second moment regularisation loss.

        Args:
            nn_output (torch.Tensor): Output from the neural network.

        Returns:
            torch.Tensor: Computed regularisation loss.
        """

        E = torch.sum((torch.abs(filter)**2), dim = -1) + self.eps

        mu = torch.sum((self.reg_idx*filter.abs()**2), dim = -1) / E

        return(mu.squeeze())

class second_moment_regularisation:
    """
    Regularisation class to compute second moment regularisation loss.
    """
    def __init__(self, n_samples: int, device='cuda', dtype=torch.float32, eps = 1e-8):
        """
        Initialize the regularisation with a weight.

        """
        
        self.multi_flag = False
        self.eps = eps
        self.n_samples = n_samples
        self.device = device
        self.dtype = dtype
        self.reg_idx = ((torch.arange(n_samples, device=device, dtype=dtype))).view(1,1,-1)

        self.loss_norm = self.compute_loss(torch.randn(1,1,self.n_samples, device=device, dtype=dtype))
        if (not torch.isfinite(self.loss_norm)) or (self.loss_norm <= self.eps):
            self.loss_norm = torch.tensor(1.0, device=device, dtype=dtype)

    def compute_loss(self, filter: torch.Tensor) -> torch.Tensor:
        """
        Compute the second moment regularisation loss.

        Args:
            nn_output (torch.Tensor): Output from the neural network.

        Returns:
            torch.Tensor: Computed regularisation loss.
        """
        E = torch.sum((torch.abs(filter)**2), dim = -1) + self.eps# Compute the energy of the filters  

        mu = torch.sum((self.reg_idx*filter.abs()**2), dim = -1)/E # Compute the 

        sig = torch.sum((self.reg_idx - mu[..., None])**2 * filter.abs()**2, dim = -1)/E

        return(sig.squeeze())
    

class source_flatness:
    
    def __init__(self, n_samples: int, device='cuda', dtype=torch.float32, eps = 1e-8):
        self.multi_flag = True
        self.eps = eps
        self.n_samples = n_samples
        self.reg_idx = ((torch.arange(n_samples, device=device, dtype=dtype))).view(1,1,-1)
        self.loss_norm = self.compute_loss(torch.randn(1,1,self.n_samples, device=device, dtype=dtype))

    def _compute_flatness(self, filters: torch.Tensor) -> torch.Tensor:

        # filters shape: [batch_size, n_filters, n_samples]
        # Compute per-filter spectra, combine filters in the frequency domain, then compute flatness.
        spectra = torch.fft.rfft(filters, dim=-1)
        combined_spectrum = torch.sum(spectra, dim=1)
        magnitude = torch.abs(combined_spectrum) + self.eps

        geometric_mean = torch.exp(torch.mean(torch.log(magnitude), dim=-1))
        arithmetic_mean = torch.mean(magnitude, dim=-1)
        flatness = geometric_mean / arithmetic_mean
        flatness = torch.clamp(flatness, 0.0, 1.0)

        return flatness.squeeze()

    
    def compute_loss(self, filters: torch.Tensor) -> torch.Tensor:
        return 1 - self._compute_flatness(filters)