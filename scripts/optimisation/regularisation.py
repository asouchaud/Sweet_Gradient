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
    
    def __init__(
        self,
        n_samples: int,
        device='cuda',
        dtype=torch.float32,
        eps = 1e-8,
        delay_samples=None,
        spk_pos=None,
        steer_point=None,
        steer_dir=None,
        fs=48000.0,
        c=343.0,
    ):
        self.multi_flag = True
        self.eps = eps
        self.n_samples = n_samples
        self.device = device
        self.dtype = dtype
        self.delay_samples = None
        if delay_samples is not None:
            self.set_relative_delays(delay_samples, device=device, dtype=dtype)
            n_filters_norm = int(self.delay_samples.numel())
        elif spk_pos is not None:
            self.set_geometry_delays(
                spk_pos=spk_pos,
                steer_point=steer_point,
                steer_dir=steer_dir,
                fs=fs,
                c=c,
                device=device,
                dtype=dtype,
            )
            n_filters_norm = int(self.delay_samples.numel())
        else:
            n_filters_norm = 1
        self.loss_norm = self.compute_loss(torch.randn(1, n_filters_norm, self.n_samples, device=device, dtype=dtype))

    def set_relative_delays(self, delay_samples, device=None, dtype=None):
        """
        Set per-source relative delays (in samples) used for coherent summation.
        Positive delay means that source is delayed in time.
        """
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        self.delay_samples = torch.as_tensor(delay_samples, device=target_device, dtype=target_dtype).reshape(-1)

    def _get_unit_steering_direction(self, spk_pos: torch.Tensor, steer_point=None, steer_dir=None) -> torch.Tensor:
        if (steer_point is None) and (steer_dir is None):
            raise ValueError("Provide either steer_point or steer_dir when using geometry-based delays.")

        if (steer_point is not None) and (steer_dir is not None):
            raise ValueError("Provide only one of steer_point or steer_dir, not both.")

        if steer_dir is not None:
            direction = torch.as_tensor(steer_dir, device=spk_pos.device, dtype=spk_pos.dtype).reshape(-1)
        else:
            array_centre = torch.mean(spk_pos, dim=0)
            target = torch.as_tensor(steer_point, device=spk_pos.device, dtype=spk_pos.dtype).reshape(-1)
            direction = target - array_centre

        norm = torch.linalg.norm(direction) + self.eps
        if norm <= self.eps:
            raise ValueError("Steering direction magnitude is near zero; check steer_point/steer_dir.")

        return direction / norm

    def set_geometry_delays(self, spk_pos, steer_point=None, steer_dir=None, fs=48000.0, c=343.0, device=None, dtype=None):
        """
        Compute relative delays (in samples) from loudspeaker geometry and steering direction.
        - steer_point: target point in space (same dimensionality as spk_pos)
        - steer_dir: steering direction vector (same dimensionality as spk_pos)
        """
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        spk_pos_t = torch.as_tensor(spk_pos, device=target_device, dtype=target_dtype)
        if spk_pos_t.ndim != 2:
            raise ValueError(f"spk_pos must have shape [n_sources, n_dims], got {tuple(spk_pos_t.shape)}.")

        unit_dir = self._get_unit_steering_direction(spk_pos_t, steer_point=steer_point, steer_dir=steer_dir)
        if unit_dir.numel() != spk_pos_t.shape[1]:
            raise ValueError(
                f"Steering vector has {unit_dir.numel()} dims, but spk_pos has {spk_pos_t.shape[1]} dims."
            )

        # Use centred geometry so delays depend on array shape/orientation, not global origin.
        rel_pos = spk_pos_t - torch.mean(spk_pos_t, dim=0, keepdim=True)
        path_projection_m = rel_pos @ unit_dir
        tau_s = path_projection_m / float(c)
        delay_samples = tau_s * float(fs)
        delay_samples = delay_samples - torch.min(delay_samples)
        self.delay_samples = delay_samples.reshape(-1)

    def _compute_flatness(self, filters: torch.Tensor) -> torch.Tensor:

        # filters shape: [batch_size, n_filters, n_samples]
        # Compute per-filter spectra, apply optional inter-source phase delays,
        # combine filters in the frequency domain, then compute flatness.
        spectra = torch.fft.rfft(filters, dim=-1)
        if self.delay_samples is not None:
            n_filters = filters.shape[1]
            if self.delay_samples.numel() != n_filters:
                raise ValueError(
                    f"delay_samples has {self.delay_samples.numel()} entries, but received {n_filters} filters."
                )
            freq_norm = torch.fft.rfftfreq(filters.shape[-1], d=1.0).to(device=filters.device, dtype=filters.dtype)
            omega = 2.0 * torch.pi * freq_norm
            phase = torch.exp(-1j * omega[None, :] * self.delay_samples.to(device=filters.device, dtype=filters.dtype)[:, None])
            spectra = spectra * phase[None, :, :]
        combined_spectrum = torch.sum(spectra, dim=1)
        magnitude = torch.abs(combined_spectrum) + self.eps

        geometric_mean = torch.exp(torch.mean(torch.log(magnitude), dim=-1))
        arithmetic_mean = torch.mean(magnitude, dim=-1)
        flatness = geometric_mean / arithmetic_mean
        flatness = torch.clamp(flatness, 0.0, 1.0)

        return flatness.squeeze()

    
    def compute_loss(self, filters: torch.Tensor) -> torch.Tensor:
        return 1 - self._compute_flatness(filters)