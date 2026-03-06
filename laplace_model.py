import torch
import torch.nn.functional as F
from model import GarmentClassifier, noise


def laplace_noise(x, noise_std, length):
    if noise_std == 0:
        return torch.zeros(x.size(0), length, device=x.device, dtype=x.dtype)

    norms = torch.sum(x ** 2, dim=1)
    norms_augm = noise_std * (norms + 1)
    norms_augm_dupl = norms_augm.repeat_interleave(length)

    loc = torch.zeros_like(norms_augm_dupl)
    scale = norms_augm_dupl
    laplace = torch.distributions.Laplace(loc, scale)
    sampled_noise = laplace.sample()

    return sampled_noise.reshape(-1, length)


class FlexibleGarmentClassifier(GarmentClassifier):
    """
    noise_mode:
        - 'none'    : no noise, standard SGD
        - 'prop'    : original proportional Gaussian noise from models.py
        - 'laplace' : Laplace-distributed noise
    """
    def __init__(self, noise_mode="none", noise_std=0.0):
        super().__init__(noise_type="prop", noise_std=noise_std)
        self.noise_mode = noise_mode

    def _get_noise(self, x, length):
        if self.noise_mode == "none":
            return torch.zeros(x.size(0), length, device=x.device, dtype=x.dtype)
        elif self.noise_mode == "prop":
            return noise(x, "prop", self.noise_std, length)
        elif self.noise_mode == "laplace":
            return laplace_noise(x, self.noise_std, length)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)

        x = self.bn1(x)
        x = F.relu(self.fc1(x)) + self._get_noise(x, 120)

        x = self.bn2(x)
        x = F.relu(self.fc2(x)) + self._get_noise(x, 120)

        x = self.bn3(x)
        x = self.fc3(x) + self._get_noise(x, 100)

        return x
