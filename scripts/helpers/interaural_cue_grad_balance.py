import torch


class InterauralCueGradBalance(torch.autograd.Function):
    """
    Identity on the forward pass. On the backward pass, rescales gradients w.r.t.
    ITD and ILD so both have the same Frobenius norm (the mean of the two norms
    before scaling). That equalises how strongly each cue drives optimisation
    through the shared peripheral pathway.

    A plain register_hook on each tensor cannot do this reliably: hooks fire
    separately and the first branch has already back-propagated before the
    second gradient is known. A single autograd.Function sees both incoming
    grads in one backward call.
    """

    @staticmethod
    def forward(ctx, itd: torch.Tensor, ild: torch.Tensor):
        return itd, ild

    @staticmethod
    def backward(ctx, grad_itd, grad_ild):
        if grad_itd is None and grad_ild is None:
            return None, None
        if grad_itd is None:
            return None, grad_ild
        if grad_ild is None:
            return grad_itd, None

        eps = torch.finfo(grad_itd.dtype).eps ** 0.5
        n_itd = torch.linalg.vector_norm(grad_itd).clamp_min(eps)
        n_ild = torch.linalg.vector_norm(grad_ild).clamp_min(eps)
        target = (n_itd + n_ild) * 0.5
        return grad_itd * (target / n_itd), grad_ild * (target / n_ild)
