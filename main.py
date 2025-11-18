import math

import torch
from time import time
from torch.utils.tensorboard import SummaryWriter
from src.noise import make_noisy_model
from src.utils import create_run_name
from src.data import dataset_creator
from src.loss_function import compute_01_loss
from src.model import *
from src.seed import set_seed
from src.training import train_one_epoch
from src.variance_provider import IsotropicVarianceProvider, AdamSqGradsVarianceProvider, InvAdamSqGradsVarianceProvider
from tqdm import trange


VARIANCE_PROVIDER_FACTORIES = {
    "inv_sq_grads": lambda optimizer, cfg: InvAdamSqGradsVarianceProvider(optimizer, 0.0, cfg['noise_std']**2),
    "sq_grads": lambda optimizer, cfg: AdamSqGradsVarianceProvider(optimizer, 0.0, cfg['noise_std']**2),
    "isotropic": lambda optimizer, cfg: IsotropicVarianceProvider(cfg['noise_std']**2),
}


class NoiseAnihilator:
    def __init__(self, total_steps: int, alpha: float):
        self.total_steps = total_steps
        self.current_step = 0
        self.alpha = alpha

    def step(self):
        self.current_step = min(self.current_step + 1, self.total_steps - 1)

    def get_noise_scale(self):
        return (self.total_steps - 1 - self.current_step) / (self.total_steps - 1)
        # return 1 / math.sqrt(1 + self.alpha * self.current_step)


def main(cfg):
    set_seed(cfg['seed'])
    writer = SummaryWriter(f"./logs/{create_run_name(cfg)}")

    print("\nStandard training loop initialized.\n")

    # We initialize the various component for our training.
    training_loader, validation_loader, classes = dataset_creator()
    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    if cfg['weights_init'] == "zero":
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                continue

            if hasattr(module, "weight"):
                torch.nn.init.zeros_(module.weight)
            if hasattr(module, "bias"):
                torch.nn.init.zeros_(module.bias)

    if cfg['noise_std'] is not None:
        covariance_mode = cfg['covariance_mode']
        if covariance_mode not in VARIANCE_PROVIDER_FACTORIES:
            raise RuntimeError(f"Invalid covariance mode '{covariance_mode}'. Options are: {list(VARIANCE_PROVIDER_FACTORIES)}")
        var_provider = VARIANCE_PROVIDER_FACTORIES[covariance_mode](optimizer, cfg)
        if cfg['noise_anihilation'] == "bayesian":
            num_batches = 98 * cfg['n_epochs']
            noise_anihilator = NoiseAnihilator(num_batches, (cfg['noise_reduction_at_end']**2 - 1) / num_batches)
            # noise_anihilator = NoiseAnihilator(98 * cfg['n_epochs'], 1.0)
            # noise_anihilator = NoiseAnihilator(98 * cfg['n_epochs'], 0.6)
        else:
            noise_anihilator = NoiseAnihilator(98 * cfg['n_epochs'], 0)
        noise_handle = make_noisy_model(model, var_provider, noise_anihilator)
    else:
        noise_handle = None
        noise_anihilator = None

    # If GPU is available: make use of it
    if torch.cuda.is_available():
        model = model.cuda()

    best_vloss = torch.inf

    # t_init = time()
    for epoch in trange(cfg['n_epochs']):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        train_one_epoch(training_loader, optimizer, noise_anihilator, model, loss_fn)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        avg_tloss, avg_vloss = compute_01_loss(model, training_loader, validation_loader)
        # cur_time = round(time() - t_init, 2)
        # print('Epoch number: {}; Train 0-1 loss: {}; Valid 0-1 loss: {}; Running time: {}s.'.format(
        #     epoch + 1, round(avg_tloss.item(), 4), round(avg_vloss.item(), 4), cur_time))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        writer.add_scalar('loss/train', avg_tloss.item(), epoch)
        writer.add_scalar('loss/valid', avg_vloss.item(), epoch)

    if noise_handle is not None:
        noise_handle.remove()
    writer.close()
    print("\nTraining finished.")


if __name__ == '__main__':
    config = {
        'project_name': 'SSGD',
        'seed': 20250729,
        'dataset': 'CIFAR100',
        'noise_std': 0.02,
        'covariance_mode': "isotropic",  # 'isotropic', 'sq_grads', 'inv_sq_grads'
        'noise_anihilation': "bayesian",
        'noise_reduction_at_end': 10.0,
        'weights_init': "zero",
        'n_epochs': 40,
        'lr': 1e-3,
    }

    main(config)
