import math
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


# Bineta's method!
class GradsStatsVarianceProvider:
    def __init__(self, optimizer: torch.optim.Optimizer, var_scalar: float = 1.0, eps=1e-8):
        self.var_scalar = var_scalar
        self.eps = eps
        self.optimizer = optimizer

    def __call__(self, param: nn.Parameter) -> torch.Tensor:
        state = self.optimizer.state.get(param, None)
        if state is None:
            return param.new_full(param.shape, 0.0)

        v = state.get("noise_var", None)
        if v is None:
            return param.new_full(param.shape, 0.0)

        v = param.new_full(param.shape, v)

        if self.var_scalar != 1.0:
            v = v * self.var_scalar

        return v


# for `GradsStatsVarianceProvider`
class GradsStatsHook:
    """
    Version locale : chaque paramètre reçoit sa propre variance et son propre sigma.
    Cela permet d'injecter un bruit proportionnel à la stabilité locale des gradients.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, alpha=0.1, sigma_min=1e-3, beta_var=0.99, history_size=5, gamma=1.0):
        self.optimizer = optimizer
        self.alpha = alpha
        self.sigma_min = sigma_min
        self.beta_var = beta_var
        self.history_size = history_size
        self.gamma = gamma

        self._init = False

        optimizer.register_step_post_hook(self.step_post_hook)

    @torch.no_grad()
    def step_post_hook(self, *args, **kwargs):
        # EMA locale stockée paramètre par paramètre
        if not self._init:
            for group in self.optimizer.param_groups:
                group["alpha"] = self.alpha
                group["sigma_min"] = self.sigma_min
                group["beta_var"] = self.beta_var
                group["history_size"] = self.history_size
                group["gamma"] = self.gamma

                for p in group['params']:
                    self.optimizer.state[p]['grad_history'] = []  # HISTORIQUE des gradients
                    self.optimizer.state[p]['var_ema'] = 0.0
                    self.optimizer.state[p]['last_sigma'] = 0.0
                    self.optimizer.state[p]['last_variance'] = 0.0

            self._init = True

        for group in self.optimizer.param_groups:
            alpha = group['alpha']
            sigma_min = group['sigma_min']
            beta = group['beta_var']
            history_size = group['history_size']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                grad = p.grad.detach().clone()

                history = state['grad_history']
                history.append(grad)
                if len(history) > history_size:
                    history.pop(0)

                if len(history) >= 2:
                    grad_stack = torch.stack(history)
                    temporal_variance = torch.var(grad_stack, dim=0)
                    var_emp = torch.mean(temporal_variance).item()
                else:
                    var_emp = 0.0

                var_ema_prev = state['var_ema']
                var_ema = beta * var_ema_prev + (1 - beta) * var_emp
                state['var_ema'] = var_ema

                # Sigma local
                sigma = alpha * math.sqrt(var_ema + 1e-12) + sigma_min
                variance_base = sigma ** 2
                state['last_variance'] = variance_base
                variance_scaled = gamma * variance_base
                sigma_final = math.sqrt(variance_scaled + 1e-12)

                state['last_sigma'] = sigma_final
                state['noise_var'] = variance_scaled


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


# KaimingVarianceProvider and XavierVarianceProvider attempt to imitate


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
