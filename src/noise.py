import functools
import torch
import torch.nn as nn
from src.variance_provider import ParamVarianceProvider
from typing import Optional, Tuple


class ModelRemovableHandle:
    def __init__(self, handles: list):
        self.handles = handles

    def remove(self):
        for handle in self.handles:
            handle.remove()


def make_noisy_model(model: nn.Module, var_provider: ParamVarianceProvider, noise_anihilator) -> ModelRemovableHandle:
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hook = functools.partial(conv2d_hook, var_provider=var_provider, noise_anihilator=noise_anihilator)
            handles.append(module.register_forward_hook(hook))
        elif isinstance(module, nn.Linear):
            hook = functools.partial(linear_hook, var_provider=var_provider, noise_anihilator=noise_anihilator)
            handles.append(module.register_forward_hook(hook))
        # elif isinstance(module, nn.BatchNorm1d):
        #     hook = functools.partial(batch_norm1d_hook, var_provider=var_provider, noise_anihilator=noise_anihilator)
        #     handles.append(module.register_forward_hook(hook))

    return ModelRemovableHandle(handles)


def conv2d_hook(
    module: nn.Conv2d,
    args: Tuple[torch.Tensor],
    mean: torch.Tensor,
    *,
    var_provider: ParamVarianceProvider,
    noise_anihilator,
) -> Optional[torch.Tensor]:
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

    std = var.clamp_min(1e-12).sqrt()  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std

    noise_scale = noise_anihilator.get_noise_scale()
    y = mean + eps * noise_scale
    return y


def linear_hook(
    module: nn.Linear,
    args: Tuple[torch.Tensor],
    mean: torch.Tensor,
    *,
    var_provider: ParamVarianceProvider,
    noise_anihilator,
) -> Optional[torch.Tensor]:
    if not module.training:
        return None

    x = args[0]

    weight_var = var_provider(module.weight)  # (out, in)
    bias_var = var_provider(module.bias) if module.bias is not None else None  # (out,)

    var = torch.nn.functional.linear(x*x, weight_var, bias_var)  # (B, out)

    std = var.clamp_min(1e-12).sqrt()  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std

    noise_scale = noise_anihilator.get_noise_scale()
    y = mean + eps * noise_scale
    return y


def batch_norm1d_hook(
    module: nn.BatchNorm1d,
    _args: Tuple[torch.Tensor],
    mean: torch.Tensor,
    *,
    var_provider: ParamVarianceProvider,
    noise_anihilator,
) -> Optional[torch.Tensor]:
    if not module.training:
        return None

    weight = module.weight.view(1, -1, *[1] * (mean.dim() - 2))
    if module.bias is not None:
        bias = module.bias.view(1, -1, *[1] * (mean.dim() - 2))
    else:
        bias = 0.

    x_norm = (mean - bias) / (weight + 1e-12)

    weight_var = var_provider(module.weight)
    bias_var = var_provider(module.bias) if module.bias is not None else 0.

    wv = weight_var.view(1, -1, *[1] * (mean.dim() - 2))
    bv = bias_var.view(1, -1, *[1] * (mean.dim() - 2)) if module.bias is not None else 0.

    var = (x_norm * x_norm) * wv + bv

    std = var.clamp_min(1e-12).sqrt()  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std

    noise_scale = noise_anihilator.get_noise_scale()
    y = mean + eps * noise_scale
    return y
