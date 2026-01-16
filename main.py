import contextlib
import pathlib
import torch
from torch.utils.tensorboard import SummaryWriter
from src.noise import NoiseHook
from src.noise_scheduler import PartialNoiseScheduler, LinearNoiseScheduler
from src.utils import create_run_name
from src.data import dataset_creator
from src.loss_function import compute_01_loss
from src.model import *
from src.seed import set_seed
from src.training import train_one_epoch
from src.variance_provider import (
    GradsStatsHook, GradsStatsVarianceProvider, IsotropicVarianceProvider, AdamSqGradsVarianceProvider,
    InvAdamSqGradsVarianceProvider, KaimingVarianceProvider, SoftmaxAdamSqGradsVarianceProvider, XavierVarianceProvider,
)
from tqdm import trange


VARIANCE_PROVIDER_FACTORIES = {
    "isotropic": lambda optimizer, cfg: IsotropicVarianceProvider(cfg['noise_std']**2),
    "sq_grads": lambda optimizer, cfg: AdamSqGradsVarianceProvider(optimizer, cfg['noise_std']**2),
    "inv_sq_grads": lambda optimizer, cfg: InvAdamSqGradsVarianceProvider(optimizer, cfg['noise_std']**2),
    "softmax_sq_grads": lambda optimizer, cfg: SoftmaxAdamSqGradsVarianceProvider(optimizer, cfg['noise_std']**2),
    "bineta": lambda optimizer, cfg: GradsStatsVarianceProvider(optimizer, cfg['noise_std'] ** 2),
    "kaiming": lambda optimizer, cfg: KaimingVarianceProvider(cfg['noise_std']**2),
    "xavier": lambda optimizer, cfg: XavierVarianceProvider(cfg['noise_std']**2),
}


NOISE_SCHEDULER_FACTORIES = {
    None: lambda *_args: None,
    "linear": lambda total_steps, cfg: LinearNoiseScheduler(total_steps),
    "partial": lambda total_steps, cfg: PartialNoiseScheduler(total_steps, cfg.get('noise_scheduler_start_step_ratio', 0.0), cfg.get('noise_scheduler_end_step_ratio', 1.0)),
}


def main(cfg):
    set_seed(cfg['seed'])
    writer = SummaryWriter(str(cfg['log_dir'] / create_run_name(cfg)))

    print("\nStandard training loop initialized.\n")

    training_loader, validation_loader, classes = dataset_creator()
    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    optimizer_hook = get_optimizer_hook(optimizer, cfg)

    noise_hook, noise_scheduler = get_noise_hook(model, optimizer, training_loader, cfg)

    if torch.cuda.is_available():
        model = model.cuda()

    best_vloss = torch.inf

    for epoch in trange(cfg['n_epochs']):
        with noise_hook, optimizer_hook:
            model.train()
            train_one_epoch(training_loader, optimizer, noise_scheduler, model, loss_fn, cfg['num_noise_samples'])

        model.eval()
        avg_tloss, avg_vloss = compute_01_loss(model, training_loader, validation_loader)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        writer.add_scalar('loss/train', avg_tloss.item(), epoch)
        writer.add_scalar('loss/valid', avg_vloss.item(), epoch)

    writer.close()
    print("\nTraining finished.")


def get_optimizer_hook(optimizer, cfg):
    if cfg.get("covariance_mode") in ("bineta",):
        return GradsStatsHook(optimizer, alpha=0.01, sigma_min=1e-3, history_size=5, gamma=0.1)
    else:
        return contextlib.nullcontext()


def get_noise_hook(model, optimizer, training_loader, cfg):
    noise_hook = contextlib.nullcontext()
    noise_scheduler = None

    if cfg['noise_std'] is not None:
        covariance_mode = cfg['covariance_mode']
        if covariance_mode not in VARIANCE_PROVIDER_FACTORIES:
            raise RuntimeError(f"Invalid covariance mode '{covariance_mode}'. Options are: {list(VARIANCE_PROVIDER_FACTORIES)}")
        var_provider = VARIANCE_PROVIDER_FACTORIES[covariance_mode](optimizer, cfg)

        noise_scheduler_id = cfg['noise_scheduler']
        if noise_scheduler_id not in NOISE_SCHEDULER_FACTORIES:
            raise RuntimeError(f"Invalid noise scheduler '{noise_scheduler_id}'. Options are: {list(NOISE_SCHEDULER_FACTORIES)}")

        total_steps = len(training_loader) * cfg['n_epochs']
        noise_scheduler = NOISE_SCHEDULER_FACTORIES[noise_scheduler_id](total_steps, cfg)

        noise_hook = NoiseHook(model, var_provider, noise_scheduler)

    return noise_hook, noise_scheduler


if __name__ == '__main__':
    config = {
        'project_name': 'SSGD',
        'seed': 20250729,
        'dataset': 'CIFAR100',
        'noise_std': None,
        'num_noise_samples': 16,
        'covariance_mode': "bineta",  # 'isotropic', 'sq_grads', 'inv_sq_grads', 'softmax_sq_grads', 'bineta', 'kaiming', 'xavier'
        'noise_scheduler': None,  # None, 'linear', 'partial'
        # 'noise_scheduler_start_step_ratio': 0.5,
        # 'noise_scheduler_end_step_ratio': 1.0,
        'n_epochs': 120,
        'lr': 5e-4,
        'log_dir': pathlib.Path("./logs_ablation"),
    }

    num_seeds = 5
    base_seed = 20250729
    seeds = [base_seed + i for i in range(num_seeds)]

    noise_levels = 7
    scale_down_levels = 3

    for seed in seeds:
        config_seed = config.copy()
        config_seed["seed"] = seed
        for covariance_mode, base_noise_std in (
            ("isotropic", None),  # no-noise baseline
            ("isotropic", 0.01),
            ("sq_grads", 0.01),
            ("inv_sq_grads", 0.01),
            ("bineta", 7.5e+2**(1/2)),  # approximate same-scale as other methods
        ):
            if base_noise_std is None:
                config_specific = config_seed.copy()
                config_specific["covariance_mode"] = covariance_mode
                config_specific["noise_std"] = None
                main(config_specific)
                continue

            # sigma scaling factors follow a geometric series
            noise_std_pre_scale = base_noise_std / 2**scale_down_levels
            for scale_up_levels in range(noise_levels):
                sigma = noise_std_pre_scale * 2**scale_up_levels
                config_specific = config_seed.copy()
                config_specific["covariance_mode"] = covariance_mode
                config_specific["noise_std"] = sigma
                main(config_specific)
