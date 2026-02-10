import torch

def select_regularistion(reg: str, n_samples: int, device='cuda', dtype=torch.float32):

    if reg == 0:
        return None
    elif reg == 1:
        return first_moment_regularisation(n_samples=n_samples, device=device, dtype=dtype)
    elif reg == 2:
        return second_moment_regularisation(n_samples=n_samples, device=device, dtype=dtype)
    else:
        return None

    
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
    def __init__(self, n_samples: int, device='cuda', dtype=torch.float32):
        """
        Initialize the regularisation with a weight.
        """
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

        mu = torch.sum((self.reg_idx*filter.abs()**2), dim = -1)

        return(mu.squeeze())

class second_moment_regularisation:
    """
    Regularisation class to compute second moment regularisation loss.
    """
    def __init__(self, n_samples: int, device='cuda', dtype=torch.float32):
        """
        Initialize the regularisation with a weight.

        """
        self.n_samples = n_samples
        self.reg_idx = ((torch.arange(n_samples, device=device, dtype=dtype))).view(1,1,-1)

        self.loss_norm = self.compute_loss(torch.randn(1,1,self.n_samples, device=device, dtype=dtype))

    def compute_loss(self, filter: torch.Tensor) -> torch.Tensor:
        """
        Compute the second moment regularisation loss.

        Args:
            nn_output (torch.Tensor): Output from the neural network.

        Returns:
            torch.Tensor: Computed regularisation loss.
        """
        E = torch.sum((torch.abs(filter)**2), dim = -1) # Compute the energy of the filters  

        mu = torch.sum((self.reg_idx*filter.abs()**2), dim = -1)/E # Compute the 

        sig = torch.sum((self.reg_idx - mu[..., None])**2 * filter.abs()**2, dim = -1)/E

        return(sig.squeeze())
    

