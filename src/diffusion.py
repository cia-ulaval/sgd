import torch
import math
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from torch import nn
from typing import Tuple
from tqdm import trange


class ForwardProcess:
    def __init__(self, betas: torch.Tensor):
        self.beta = betas

        self.alphas = 1. - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=-1)

    def get_x_t(self, x_0: torch.Tensor, t: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps_0 = torch.randn_like(x_0).to(x_0)
        alpha_bar = self.alpha_bar[t, None]
        mean = (alpha_bar ** 0.5) * x_0
        std = (1. - alpha_bar).sqrt()

        return (eps_0, mean + std * eps_0)


class ReverseProcess(ForwardProcess):
    def __init__(self, betas: torch.Tensor, model: nn.Module):
        super().__init__(betas)
        self.model = model
        self.T = len(betas) - 1

        self.sigma = (
            (1 - self.alphas)
            * (1 - torch.roll(self.alpha_bar, 1)) / (1 - self.alpha_bar)
        ) ** 0.5
        self.sigma[1] = 0.0
    
    def get_x_t_minus_one(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            t_vector = torch.full(size=(len(x_t),), fill_value=t, device=x_t.device, dtype=torch.long)
            eps = self.model(x_t, t=t_vector)

        eps *= (1 - self.alphas[t]) / ((1 - self.alpha_bar[t]) ** 0.5)
        mean =  1 / (self.alphas[t] ** 0.5) * (x_t - eps)
        if t > 1:
            return mean + self.sigma[t] * torch.randn_like(x_t)
        return mean

    def sample(self, n_samples=1, full_trajectory=False):
        # Initialize with X_T ~ N(0, I)
        device = next(self.model.parameters()).device
        x_t = torch.randn(n_samples, 2, device=device)
        trajectory = [x_t.clone()]
        
        for t in range(self.T, 0, -1):
            x_t = self.get_x_t_minus_one(x_t, t=t)
            
            if full_trajectory:
                trajectory.append(x_t.clone())
        return torch.stack(trajectory, dim=0) if full_trajectory else x_t


class NoisePredictor(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.t_encoder = nn.Linear(T, 1)

        self.model = nn.Sequential(
            nn.Linear(2 + 1, 100),   # Input: Noisy data x_t and t
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 20),
            nn.LeakyReLU(inplace=True),
            # Output: Predicted noise that was added to the original data point
            nn.Linear(20, 2),
        )

    def forward(self, x_t, t):
        # Encode the time index t as one-hot and then use one layer to encode
        # into a single value
        t_embedding = self.t_encoder(
            nn.functional.one_hot(t - 1, num_classes=self.T).to(torch.float)
        )

        inp = torch.cat([x_t, t_embedding], dim=1)
        return self.model(inp)


def train_test():
    device = "cpu"
    T = 200
    X = make_swiss_roll(device=device)[0]
    model = NoisePredictor(T=T).to(device).train()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-2, betas=(0.9, 0.999), weight_decay=1e-4)

    betas = torch.zeros(T + 1, device=device)
    betas[1:] = torch.linspace(1e-4, 2e-2, T, device=device)
    fp = ForwardProcess(betas)

    N = X.shape[0]
    for epoch in trange(5000):
        with torch.no_grad():
            # Sample random t's
            t = torch.randint(low=1, high=T + 1, size=(N,), device=device)

            # Get the noise added and the noisy version of the data using the forward
            # process given t
            eps_0, x_t = fp.get_x_t(X, t=t)
        
        # Predict the noise added to x_0 from x_t
        pred_eps = model(x_t, t)

        # Simplified objective without weighting with alpha terms (Ho et al, 2020)
        loss = torch.nn.functional.mse_loss(pred_eps, eps_0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"loss: {loss.item()}")
        if epoch % 100 == 0:
            model.eval()
            print_gif(betas, model, X, epoch)
            model.train()


@torch.no_grad()
def print_gif(betas, model, real, epoch, out_dir="gifs", n_samples=2000):
    os.makedirs(out_dir, exist_ok=True)

    rp = ReverseProcess(betas, model)
    traj = rp.sample(n_samples=n_samples, full_trajectory=True).cpu()
    real = real.cpu()
    frames = []

    xmin = min(real[:, 0].min(), traj[:, :, 0].min()).item() - 0.2
    xmax = max(real[:, 0].max(), traj[:, :, 0].max()).item() + 0.2
    ymin = min(real[:, 1].min(), traj[:, :, 1].min()).item() - 0.2
    ymax = max(real[:, 1].max(), traj[:, :, 1].max()).item() + 0.2

    for i in range(len(traj)):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(real[:, 0], real[:, 1], s=3, alpha=0.15, label="data")
        ax.scatter(traj[i, :, 0], traj[i, :, 1], s=3, alpha=0.8, label="sample")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"epoch={epoch}, step={i}")
        ax.legend(loc="upper right")

        fname = f"{out_dir}/frame_{epoch:04d}_{i:04d}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        frames.append(imageio.imread(fname))

    imageio.mimsave(f"{out_dir}/epoch_{epoch:04d}.gif", frames, duration=0.05)


def make_swiss_roll(
    n_samples: int = 1000,
    noise: float = 0.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
    device = torch.device(device)

    t = 1.5 * math.pi * (1 + 2 * torch.rand(n_samples, device=device, dtype=dtype))
    x = t * torch.cos(t)
    y = 21 * torch.rand(n_samples, device=device, dtype=dtype)
    z = t * torch.sin(t)

    # X = torch.stack([x, y, z], dim=1)
    X = torch.stack([x, z], dim=1)

    if noise > 0:
        X = X + noise * torch.randn_like(X)

    return X, t


train_test()
