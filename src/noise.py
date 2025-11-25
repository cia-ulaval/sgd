import functools
import torch
import torch.nn as nn

from src.noise_scheduler import NoiseScheduler
from src.variance_provider import ParamVarianceProvider
from typing import Optional, Tuple


class ModelRemovableHandle:
    def __init__(self, handles: list):
        self.handles = handles

    def remove(self):
        for handle in self.handles:
            handle.remove()


def make_noisy_model(
    model: nn.Module,
    var_provider: ParamVarianceProvider = None,
    noise_scheduler: NoiseScheduler = None,
) -> ModelRemovableHandle:
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hook = functools.partial(conv2d_hook, var_provider=var_provider, noise_scheduler=noise_scheduler)
            handles.append(module.register_forward_hook(hook))
        elif isinstance(module, nn.Linear):
            hook = functools.partial(linear_hook, var_provider=var_provider, noise_scheduler=noise_scheduler)
            handles.append(module.register_forward_hook(hook))

    return ModelRemovableHandle(handles)


def conv2d_hook(
    module: nn.Conv2d,
    args: Tuple[torch.Tensor],
    mean: torch.Tensor,
    *,
    var_provider: Optional[ParamVarianceProvider],
    noise_scheduler: Optional[NoiseScheduler],
) -> Optional[torch.Tensor]:
    if not module.training or var_provider is None:
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

    if noise_scheduler is not None:
        noise_scale = noise_scheduler.get_noise_scalar()
        y = mean + eps * noise_scale
    else:
        y = mean + eps

    return y


def linear_hook(
    module: nn.Linear,
    args: Tuple[torch.Tensor],
    mean: torch.Tensor,
    *,
    var_provider: Optional[ParamVarianceProvider],
    noise_scheduler: Optional[NoiseScheduler],
) -> Optional[torch.Tensor]:
    if not module.training or var_provider is None:
        return None

    x = args[0]

    weight_var = var_provider(module.weight)  # (out, in)
    bias_var = var_provider(module.bias) if module.bias is not None else None  # (out,)

    var = torch.nn.functional.linear(x*x, weight_var, bias_var)  # (B, out)

    std = var.clamp_min(1e-12).sqrt()  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std

    if noise_scheduler is not None:
        noise_scale = noise_scheduler.get_noise_scalar()
        y = mean + eps * noise_scale
    else:
        y = mean + eps

    return y
