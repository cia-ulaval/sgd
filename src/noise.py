import functools
import torch
import torch.nn as nn
from src.variance_provider import ParamVarianceProvider


class ModelRemovableHandle:
    def __init__(self, handles: list):
        self.handles = handles

    def remove(self):
        for handle in self.handles:
            handle.remove()


def make_model_noisy(model: nn.Module, var_provider: ParamVarianceProvider) -> ModelRemovableHandle:
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = functools.partial(conv2d_hook, var_provider=var_provider)
            handles.append(module.register_forward_hook(hook))
        elif isinstance(module, nn.Linear):
            hook = functools.partial(linear_hook, var_provider=var_provider)
            handles.append(module.register_forward_hook(hook))

    return ModelRemovableHandle(handles)


def conv2d_hook(module: nn.Conv2d, args, mean, *, var_provider: ParamVarianceProvider):
    if not module.training:
        return None

    x = args[0]

    weight_var = var_provider(module.weight)  # (C_out, C_in_per_group, kH, kW)
    bias_var = var_provider(module.bias) if module.bias is not None else None  # (C_out,)

    var = torch.nn.functional.conv2d(
        x*x, weight_var, bias_var,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )  # (B, C_out, H_out, W_out)

    std_out = torch.sqrt(var.clamp_min(1e-12))  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std_out

    return mean + eps


def linear_hook(module: nn.Linear, args, mean, *, var_provider: ParamVarianceProvider):
    if not module.training:
        return None

    x = args[0]

    weight_var = var_provider(module.weight)  # (out, in)
    bias_var = var_provider(module.bias) if module.bias is not None else None  # (out,)

    var = torch.nn.functional.linear(x*x, weight_var, bias_var)  # (B, out)

    std_out = torch.sqrt(var.clamp_min(1e-12))  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std_out  # broadcast over out dim

    return mean + eps
