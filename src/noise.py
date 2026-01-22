import contextlib
import functools
import torch
import torch.nn as nn
from src.noise_scheduler import NoiseScheduler
from src.variance_provider import ParamVarianceProvider
from typing import ContextManager, Optional, Tuple


class NoiseHook:
    def __init__(
        self,
        model: nn.Module,
        var_provider: ParamVarianceProvider,
        noise_scheduler: Optional[NoiseScheduler] = None,
    ):
        self.model = model
        self.var_provider = var_provider
        self.noise_scheduler = noise_scheduler
        self._handles = []

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                hook = functools.partial(conv2d_hook, var_provider=self.var_provider, noise_scheduler=self.noise_scheduler)
                self._handles.append(module.register_forward_hook(hook))
            elif isinstance(module, nn.Linear):
                hook = functools.partial(linear_hook, var_provider=self.var_provider, noise_scheduler=self.noise_scheduler)
                self._handles.append(module.register_forward_hook(hook))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def conv2d_hook(
    module: nn.Conv2d,
    args: Tuple[torch.Tensor],
    mean: torch.Tensor,
    *,
    var_provider: ParamVarianceProvider,
    noise_scheduler: Optional[NoiseScheduler],
) -> Optional[torch.Tensor]:
    x = args[0]

    weight_var = var_provider(module.weight)
    bias_var = var_provider(module.bias) if module.bias is not None else None

    var = torch.nn.functional.conv2d(
        x.square(), weight_var, bias_var,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )

    std = var.clamp_min(1e-12).sqrt()  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std

    if noise_scheduler is not None:
        noise_scale = noise_scheduler.get_noise_scalar()
        eps = eps * noise_scale

    y = mean + eps
    return y


def linear_hook(
    module: nn.Linear,
    args: Tuple[torch.Tensor],
    mean: torch.Tensor,
    *,
    var_provider: ParamVarianceProvider,
    noise_scheduler: Optional[NoiseScheduler],
) -> Optional[torch.Tensor]:
    x = args[0]

    weight_var = var_provider(module.weight)
    bias_var = var_provider(module.bias) if module.bias is not None else None

    var = torch.nn.functional.linear(x.square(), weight_var, bias_var)

    std = var.clamp_min(1e-12).sqrt()  # clamp to prevent nan grads
    eps = torch.randn_like(mean) * std

    if noise_scheduler is not None:
        noise_scale = noise_scheduler.get_noise_scalar()
        eps = eps * noise_scale

    y = mean + eps
    return y
