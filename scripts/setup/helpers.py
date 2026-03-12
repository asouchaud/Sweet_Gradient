import torch 

def math_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.nn.functional.conv1d(input, torch.flip(weight, dims = [-1]), bias, stride, padding, dilation, groups)

def true_conv1d(input, weight):
    K = weight.shape[-1]
    pad_input = torch.nn.functional.pad(input, (K-1,0))
    out = torch.nn.functional.conv1d(pad_input, torch.flip(weight, [-1]))
    
    return out

def l3_loss(input, target, reduction="mean"):
    loss = torch.abs(input - target) ** 3
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
    


