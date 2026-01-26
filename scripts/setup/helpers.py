import torch 

def math_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.nn.functional.conv1d(input, torch.flip(weight, dims = [-1]), bias, stride, padding, dilation, groups)

def l3_loss(input, target, reduction="mean"):
    loss = torch.abs(input - target) ** 3
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss