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


class ConstantVarianceProvider:
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
    def __init__(self, optimizer: torch.optim.Optimizer, default_var: float = 0.0, var_scalar: float = 1.0):
        self.optimizer = optimizer
        self.default_var = default_var
        self.var_scalar = var_scalar

    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        # 1. get adam's second order momentum
        state = self.optimizer.state.get(param, None)
        if state is None:
            return param.new_full(param.shape, self.default_var)

        v = state.get("exp_avg_sq", None)
        if v is None:
            return param.new_full(param.shape, self.default_var)

        # 2. compute variance
        if self.var_scalar != 1.0:
            v = v * self.var_scalar

        return v


class InvAdamSqGradsVarianceProvider:
    """
    Uses Adam's exp_avg_sq as the *inverse variance* for each parameter.
    If a parameter doesn't (yet) have state, falls back to default_var.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, default_var: float = 0.0, var_scalar: float = 1.0, eps=1e-8):
        self.optimizer = optimizer
        self.default_var = default_var
        self.var_scalar = var_scalar
        self.eps = eps

    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        # 1. get adam's second order momentum
        state = self.optimizer.state.get(param, None)
        if state is None:
            return param.new_full(param.shape, self.default_var)

        v = state.get("exp_avg_sq", None)
        if v is None:
            return param.new_full(param.shape, self.default_var)

        # 2. compute variance
        if self.var_scalar != 1.0:
            v = v / self.var_scalar

        v = 1 / (v + self.eps)
        return v
