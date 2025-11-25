from typing import Protocol

import torch
import torch.nn as nn


class ParamVarianceProvider(Protocol):
    """
    Callable: given a Parameter, returns a tensor of variances
    with the same shape as that Parameter.
    """
    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        ...


class IsotropicVarianceProvider:
    def __init__(self, default_var: float):
        self.default_var = float(default_var)

    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        # returns a new tensor; no sharing needed here
        return param.new_full(param.shape, self.default_var)


class AdamSqGradsVarianceProvider:
    """
    Uses Adam's exp_avg_sq as the *variance* for each parameter.
    If a parameter doesn't (yet) have state, falls back to default_var.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, var_scalar: float = 1.0):
        self.optimizer = optimizer
        self.var_scalar = var_scalar

    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        # 1. get adam's second order momentum
        state = self.optimizer.state.get(param, None)
        if state is None:
            return param.new_full(param.shape, 0.0)

        v = state.get("exp_avg_sq", None)
        if v is None:
            return param.new_full(param.shape, 0.0)

        # 2. compute variance
        v = v / v.mean()

        if self.var_scalar != 1.0:
            v = v * self.var_scalar

        return v


class InvAdamSqGradsVarianceProvider:
    """
    Uses Adam's exp_avg_sq as the *inverse variance* for each parameter.
    If a parameter doesn't (yet) have state, falls back to default_var.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, var_scalar: float = 1.0, eps=1e-8):
        self.optimizer = optimizer
        self.var_scalar = var_scalar
        self.eps = eps

    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        # 1. get adam's second order momentum
        state = self.optimizer.state.get(param, None)
        if state is None:
            return param.new_full(param.shape, 0.0)

        v = state.get("exp_avg_sq", None)
        if v is None:
            return param.new_full(param.shape, 0.0)

        # 2. compute variance
        v = 1 / (v + self.eps)
        v = v / v.mean()
        v = v.clamp_min(1.0)

        if self.var_scalar != 1.0:
            v = v * self.var_scalar

        return v


class SoftmaxAdamSqGradsVarianceProvider:
    def __init__(self, optimizer: torch.optim.Optimizer, var_scalar: float = 1.0, temperature: float = 1.0, eps=1e-8):
        self.optimizer = optimizer
        self.var_scalar = var_scalar
        self.temperature = temperature
        self.eps = eps

    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        # 1. get adam's second order momentum
        state = self.optimizer.state.get(param, None)
        if state is None:
            return param.new_full(param.shape, 0.0)

        v = state.get("exp_avg_sq", None)
        if v is None:
            return param.new_full(param.shape, 0.0)

        logits = -self.temperature * (v + self.eps).log()
        logits_exp = (logits - logits.max()).exp()
        var = logits_exp / logits_exp.mean()

        # 2. compute variance
        if self.var_scalar != 1.0:
            var = var * self.var_scalar

        return var


class KaimingVarianceProvider:
    def __init__(self, var_scalar: float = 1.0):
        self.var_scalar = var_scalar

    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        # BatchNorm weights / biases, or any 1D param:
        if param.dim() == 1:
            # Treat BN gamma/beta as deterministic: no noise.
            return torch.zeros_like(param)

        # Linear: (out_features, in_features)
        if param.dim() == 2:
            fan_in = param.size(1)

        # ConvNd: (out_channels, in_channels/groups, *kernel)
        elif param.dim() > 2:
            # numel of one filter = fan_in (since it's per-output-channel)
            fan_in = param[0].numel()

        else:
            # shouldn’t happen
            return torch.zeros_like(param)

        var = self.var_scalar * 2.0 / fan_in
        return torch.full_like(param, var)


class XavierVarianceProvider:
    def __init__(self, var_scalar: float = 1.0):
        self.var_scalar = var_scalar

    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        # 1D params (e.g. BatchNorm gamma/beta) → no noise
        if param.dim() == 1:
            return torch.zeros_like(param)

        if param.dim() == 2:  # Linear: (out, in)
            fan_in = param.size(1)
            fan_out = param.size(0)

        elif param.dim() > 2:  # ConvNd
            # Rough but standard: use PyTorch's fan calc style
            fan_in = param.shape[1:].numel()
            fan_out = param.size(0)

        else:
            return torch.zeros_like(param)

        var = self.var_scalar * 2.0 / (fan_in + fan_out)
        return torch.full_like(param, var)
