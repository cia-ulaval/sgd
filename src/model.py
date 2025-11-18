import torch
import torch.nn as nn
import torch.nn.functional as F

from src.variance_provider import ParamVarianceProvider


class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = NoisyConv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = NoisyConv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm1d(num_features=400)
        self.fc1 = NoisyLinear(400, 120)
        self.bn2 = nn.BatchNorm1d(num_features=120)
        self.fc2 = NoisyLinear(120, 120)
        self.bn3 = nn.BatchNorm1d(num_features=120)
        self.fc3 = NoisyLinear(120, 100)

    def set_variance_provider(self, variance_provider: ParamVarianceProvider):
        self.conv1.variance_provider = variance_provider
        self.conv2.variance_provider = variance_provider
        self.fc1.variance_provider = variance_provider
        self.fc2.variance_provider = variance_provider
        self.fc3.variance_provider = variance_provider

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.fc3(x)
        return x


class SimpleMLPClassifier(nn.Module):
    def __init__(self):
        super(SimpleMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NoisyLinear(nn.Linear):
    def __init__(self, in_dims, out_dims,
                 bias: bool = True,
                 base_var: float = 0.0,
                 variance_provider: ParamVarianceProvider | None = None,
                 device=None, dtype=None):
        super().__init__(in_dims, out_dims, bias=bias, device=device, dtype=dtype)
        self.base_var = float(base_var)
        self.variance_provider = variance_provider  # can be set later

    def _get_weight_var(self) -> torch.Tensor:
        if self.variance_provider is None:
            return self.weight.new_full(self.weight.shape, self.base_var)
        return self.variance_provider(self.weight)

    def _get_bias_var(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        if self.variance_provider is None:
            return self.bias.new_full(self.bias.shape, self.base_var)
        return self.variance_provider(self.bias)

    def forward(self, x):
        mean = F.linear(x, self.weight, self.bias)   # (B, out_features)

        if (self.base_var == 0.0 and self.variance_provider is None) or not self.training:
            return mean

        weight_var = self._get_weight_var()  # (out, in)
        bias_var = self._get_bias_var() if self.bias is not None else None  # (out,)

        x2 = x * x  # (B, in)
        var = F.linear(x2, weight_var, bias_var)  # (B, out)

        std_out = torch.sqrt(var.clamp_min(1e-12))  # clamp to prevent nan grads
        eps = torch.randn_like(mean) * std_out  # broadcast over out dim

        return mean + eps


class NoisyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 base_var: float = 0.0,
                 variance_provider: ParamVarianceProvider | None = None,
                 device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups,
            bias=bias, device=device, dtype=dtype
        )
        self.base_var = float(base_var)
        self.variance_provider = variance_provider

    def _get_weight_var(self) -> torch.Tensor:
        if self.variance_provider is None:
            return self.weight.new_full(self.weight.shape, self.base_var)
        return self.variance_provider(self.weight)

    def _get_bias_var(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        if self.variance_provider is None:
            return self.bias.new_full(self.bias.shape, self.base_var)
        return self.variance_provider(self.bias)

    def forward(self, x):
        mean = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if (self.base_var == 0.0 and self.variance_provider is None) or not self.training:
            return mean

        weight_var = self._get_weight_var()  # (C_out, C_in_per_group, kH, kW)
        bias_var = self._get_bias_var() if self.bias is not None else None  # (C_out,)

        x2 = x * x
        var = F.conv2d(
            x2, weight_var, bias_var,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )  # (B, C_out, H_out, W_out)

        std_out = torch.sqrt(var.clamp_min(1e-12))  # clamp to prevent nan grads
        eps = torch.randn_like(mean) * std_out

        return mean + eps
