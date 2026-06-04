import torch
import torch.nn.functional as F

def softmax_st(z, dim = -1, tau=1.0):
    y_soft = F.softmax(z / tau, dim=dim)
    index = z.max(dim=dim, keepdim=True)[1]
    y_hard = torch.zeros_like(z).scatter_(dim, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft


def math_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.nn.functional.conv1d(input, torch.flip(weight, dims = [-1]), bias, stride, padding, dilation, groups)


def causal_conv1d(input, weight):
    K = weight.shape[-1]
    pad_input = torch.nn.functional.pad(input, (K-1,0))
    out = torch.nn.functional.conv1d(pad_input, torch.flip(weight, [-1]))
    
    return out


def hilbert_transform(x):
    """
    Differentiable Hilbert transform for real input of shape (B, C, N).
    Returns the complex analytic signal of shape (B, C, N).
    """
    N = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)

    # Build the Hilbert filter mask
    h = torch.zeros(N, dtype=x.dtype, device=x.device)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    H = h.view(1, 1, N)  # Shape: [1, 1, N] to broadcast over B and C
    Xf_filtered = Xf * H

    x_analytic = torch.fft.ifft(Xf_filtered, dim=-1)
    return x_analytic  # Complex tensor: real + j*hilbert(real)


def rms(x, dim, keepdim = False):
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=keepdim))


def l3_loss(input, target, reduction="mean"):
    loss = torch.abs(input - target) ** 3
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def calculate_source_angle_2D(vec1):
    vec1_i = vec1[:, 0] + 1j * vec1[:, 1]
    theta = torch.rad2deg(torch.angle(vec1_i))%360
    return theta


def gtfb_center_frequencies(gtfb, fs, n_fft):
    """Peak-magnitude center frequency (Hz) for each GTFB channel."""
    filters = gtfb.squeeze(1) if gtfb.dim() == 3 else gtfb
    fft_freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(fs)).to(device=filters.device, dtype=filters.dtype)
    magnitude = torch.abs(torch.fft.rfft(filters, n=n_fft, dim=-1))
    return fft_freqs[torch.argmax(magnitude, dim=-1)]


def erb_frequency_bin_weights(fft_frequencies, eps=1e-12):
    """Per-FFT-bin ERB weights with sum(weights) == M (number of bins).

    Relative shape is (1/ERB) / max(1/ERB); ERB = 0.108 * freq + 24.7.
    """
    erb = 0.108 * fft_frequencies + 24.7
    inv_erb = 1.0 / erb.clamp(min=eps)
    w = inv_erb / inv_erb.max()
    m = w.numel()
    return w * (m / w.sum())


def compute_reference_bin(reference_stim, reference_hrirs, listener_indices):
    # Compute reference binaural signal (convolution of stimulus with HRIR)

    assert reference_stim.dim() == 3, "reference_stim should have shape (batch_size, 1, signal_length)"
    assert reference_stim.shape[1] == 1, "reference_stim should have a single channel dimension"
    assert reference_hrirs.dim() == 3, "reference_hrirs should have shape (n_listeners*2, 1, ir_length)"
    

    n_lstn = len(listener_indices)
    n_stim = len(reference_stim)
    hrirs = reference_hrirs[listener_indices].view(2*n_lstn, 1, -1) # Reshape to (2*n_listeners, 1, ir_len) for convolution

    reference_bin = causal_conv1d(reference_stim, hrirs).view(n_stim, n_lstn, 2, -1) # Reshape back to (n_stim, n_listeners, 2, signal_len)

    return reference_bin.view(n_stim * n_lstn, 2, -1) # Reshape to (n_stim*n_listeners, 2, signal_len) for LSD calculation
