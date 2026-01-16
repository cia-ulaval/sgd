import pathlib
import torch
from torch.utils.tensorboard import SummaryWriter
from src.noise import make_noisy_model
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
    "partial": lambda total_steps, cfg: PartialNoiseScheduler(total_steps, cfg['noise_scheduler_start_step_ratio'], cfg['noise_scheduler_end_step_ratio']),
}


def main(cfg):
    set_seed(cfg['seed'])
    writer = SummaryWriter(str(cfg['log_dir'] / create_run_name(cfg)))

    print("\nStandard training loop initialized.\n")

    # We initialize the various component for our training.
    training_loader, validation_loader, classes = dataset_creator()
    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    if cfg.get("covariance_mode") == "bineta":
        bineta_hook = GradsStatsHook(optimizer, alpha=0.01, sigma_min=1e-3, history_size=5, gamma=0.1)

    noise_handle = None
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

        noise_handle = make_noisy_model(model, var_provider, noise_scheduler)

    # If GPU is available: make use of it
    if torch.cuda.is_available():
        model = model.cuda()

    best_vloss = torch.inf

    for epoch in trange(cfg['n_epochs']):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        train_one_epoch(training_loader, optimizer, noise_scheduler, model, loss_fn, cfg['num_noise_samples'])

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        avg_tloss, avg_vloss = compute_01_loss(model, training_loader, validation_loader)

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
        'noise_std': None,
        'num_noise_samples': 128,
        'covariance_mode': "bineta",  # 'isotropic', 'sq_grads', 'inv_sq_grads', 'softmax_sq_grads', 'bineta', 'kaiming', 'xavier'
        'noise_scheduler': None,  # None, 'linear', 'partial'
        # 'noise_scheduler_start_step_ratio': 0.5,
        # 'noise_scheduler_end_step_ratio': 1.0,
        'n_epochs': 120,
        'lr': 5e-4,
        'log_dir': pathlib.Path("./logs_ablation"),
    }

    noise_levels = 7
    scale_down_levels = 3
    for covariance_mode, base_noise_std in (
        ("isotropic", None),
        ("isotropic", 0.01),
        ("sq_grads", 0.01),
        ("inv_sq_grads", 0.01),
        ("bineta", 7.5e+2**(1/2)),
    ):
        if base_noise_std is None:
            config_specific = config.copy()
            config_specific["covariance_mode"] = covariance_mode
            config_specific["noise_std"] = None
            main(config_specific)
            continue

        noise_std_pre_scale = base_noise_std / 2**scale_down_levels
        for scale_up_levels in range(noise_levels):
            config_specific = config.copy()
            config_specific["covariance_mode"] = covariance_mode
            config_specific["noise_std"] = noise_std_pre_scale * 2**scale_up_levels
            main(config_specific)
