import contextlib
import pathlib
from typing import Generator, Optional, Tuple
import math
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from src.noise import NoiseHook
from src.noise_scheduler import PartialNoiseScheduler, LinearNoiseScheduler, TrainValidDiffNoiseScheduler, TrainValidDiffStaticPidNoiseScheduler
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
    "train_valid_differential": lambda total_steps, cfg: TrainValidDiffNoiseScheduler(cfg.get("noise_scheduler_gamma", 1.0)),
    "train_valid_pid": lambda total_steps, cfg: TrainValidDiffStaticPidNoiseScheduler(
        a_bounds=cfg.get("noise_scheduler_a_bounds", (-10, 10)),
        b_bounds=cfg.get("noise_scheduler_b_bounds", (-10, 10)),
        c_bounds=cfg.get("noise_scheduler_c_bounds", (-10, 10)),
        sigma_bounds=cfg.get("noise_scheduler_sigma_bounds", (-10, 10)),
        n_startup_trials=cfg.get("noise_scheduler_n_startup_trials", 10),
        seed=cfg["seed"],
        log_dir=cfg["log_dir"],
    ),
}


def ablate_covariance_modes():
    logdir = pathlib.Path("./logs_ablation")
    config = create_base_config(logdir)

    num_seeds = 5
    noise_std_levels = 7

    for seed in seeds(num_seeds):
        config_seed = config.copy()
        config_seed["seed"] = seed
        for covariance_mode, base_noise_std in covariance_modes():
            if base_noise_std is None:
                config_specific = config_seed.copy()
                config_specific["covariance_mode"] = covariance_mode
                config_specific["noise_std"] = None
                do_one_run(config_specific)
                continue

            for sigma in sigmas(base_noise_std, noise_std_levels):
                config_specific = config_seed.copy()
                config_specific["covariance_mode"] = covariance_mode
                config_specific["noise_std"] = sigma
                do_one_run(config_specific)


def ablate_num_samples():
    logdir = pathlib.Path("./logs_ablation_num_samples")
    config = create_base_config(logdir)

    num_seeds = 5
    num_samples_levels = 9  # 1 to 256
    num_samples_batch_max_level = 6  # 64
    noise_std_l5_scale = 4

    for seed in seeds(num_seeds):
        config_seed = config.copy()
        config_seed["seed"] = seed
        for covariance_mode, base_noise_std in covariance_modes():
            if base_noise_std is None:
                config_specific = config_seed.copy()
                config_specific["num_noise_samples_batch"] = 1
                config_specific["num_noise_samples_accumulation"] = 1
                config_specific["covariance_mode"] = covariance_mode
                config_specific["noise_std"] = None
                do_one_run(config_specific)
                continue

            for num_samples_level in range(num_samples_levels):
                num_samples_batch = 2 ** min(num_samples_level, num_samples_batch_max_level)
                num_samples_accumulation = 2 ** max(0, num_samples_level - num_samples_batch_max_level)

                config_specific = config_seed.copy()
                config_specific["num_noise_samples_batch"] = num_samples_batch
                config_specific["num_noise_samples_accumulation"] = num_samples_accumulation
                config_specific["covariance_mode"] = covariance_mode
                config_specific["noise_std"] = base_noise_std * noise_std_l5_scale
                do_one_run(config_specific)


def ablate_train_valid_differential():
    logdir = pathlib.Path("./logs_ablation_train_valid_differential")
    config = create_base_config(logdir)
    config["num_noise_samples_batch"] = 1
    config["num_noise_samples_accumulation"] = 1
    config["covariance_mode"] = "isotropic"

    num_seeds = 5
    base_noise_std = 0.01
    noise_std_levels = 7

    for seed in seeds(num_seeds):
        config_seed = config.copy()
        config_seed["seed"] = seed

        for sigma in reversed((None, *geometric_range(0.01, 0.08, 8))):
            config_specific = config_seed.copy()
            config_specific["noise_std"] = sigma
            if sigma is not None:
                for gamma in reversed((0.0, *geometric_range(1.0, 8.0, 8))):
                    config_specific["noise_scheduler"] = "train_valid_differential"
                    config_specific["noise_scheduler_gamma"] = gamma
                    do_one_run(config_specific)
            else:
                do_one_run(config_specific)

    print_results_table(logdir)


def ablate_train_valid_diff_pid():
    logdir = pathlib.Path("./logs_ablation_train_valid_diff_pid")
    config = create_base_config(logdir)

    num_iterations = 100
    config["num_noise_samples_batch"] = 1
    config["num_noise_samples_accumulation"] = 1
    config["covariance_mode"] = "isotropic"
    num_seeds = 5

    for seed in seeds(num_seeds):
        for _ in range(num_iterations):
            config_seed = config.copy()
            config_seed["seed"] = seed

            config_scheduler = config_seed.copy()
            config_scheduler["noise_scheduler"] = "train_valid_pid"
            config_scheduler["noise_scheduler_a_bounds"] = (-10, 10)
            config_scheduler["noise_scheduler_b_bounds"] = (-10, 10)
            config_scheduler["noise_scheduler_c_bounds"] = (-10, 10)
            config_scheduler["noise_scheduler_sigma_bounds"] = (-10, 10)
            config_scheduler["noise_scheduler_n_startup_trials"] = 10
            config_scheduler["num_iterations"] = num_iterations
            do_one_run(config_scheduler)

    print_results_table(logdir)


def create_base_config(log_dir: pathlib.Path):
    return {
        'project_name': 'SSGD',
        'seed': 20250729,
        'dataset': 'CIFAR100',
        'noise_std': None,
        'num_noise_samples_batch': 64,
        'num_noise_samples_accumulation': 2,
        'covariance_mode': "bineta",  # 'isotropic', 'sq_grads', 'inv_sq_grads', 'softmax_sq_grads', 'bineta', 'kaiming', 'xavier'
        'noise_scheduler': None,  # None, 'linear', 'partial'
        # 'noise_scheduler_start_step_ratio': 0.5,
        # 'noise_scheduler_end_step_ratio': 1.0,
        'n_epochs': 120,
        'lr': 5e-4,
        'batch_size': 512,
        'log_dir': log_dir,
    }


def seeds(num_seeds: int, base_seed: int = 20250729) -> Generator[int, None, None]:
    for i in range(num_seeds):
        yield base_seed + i


def sigmas(base_noise: float, noise_levels: int, base: float = 2.0) -> Generator[float, None, None]:
    # sigma scaling factors follow a geometric series
    noise_std_scale_down_levels = 3
    for noise_std_scale_up_levels in range(noise_levels):
        sigma = base_noise * base**(noise_std_scale_up_levels - noise_std_scale_down_levels)
        yield sigma


def geometric_range(min: float, max: float, steps: int):
    min_log = math.log(min)
    max_log = math.log(max)
    last_i = steps - 1
    delta_log = max_log - min_log
    for i in range(steps):
        current_ratio = i / last_i
        current_log = min_log + delta_log * current_ratio
        yield math.exp(current_log)


def covariance_modes() -> Generator[Tuple[str, Optional[float]], None, None]:
    yield from (
        ("isotropic", None),  # no-noise baseline
        ("isotropic", 0.01),
        ("sq_grads", 0.01),
        ("inv_sq_grads", 0.01),
        ("bineta", 7.5e+2**(1/2)),  # approximate same-scale as other methods
    )


def do_one_run(cfg):
    set_seed(cfg['seed'])
    run_dir = cfg['log_dir'] / create_run_name(cfg)
    writer = SummaryWriter(str(run_dir))
    best_loss_path = run_dir / "best_loss.json"

    if torch.cuda.is_available:
        device = "cuda:1"
    else:
        device = "cpu"

    print("\nStandard training loop initialized.\n")

    training_loader, validation_loader, test_loader, classes = dataset_creator(batch_size=cfg["batch_size"])
    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    optimizer_hook = get_optimizer_hook(optimizer, cfg)

    total_steps = len(training_loader) * cfg['n_epochs']
    noise_hook, noise_scheduler = get_noise_hook(model, optimizer, total_steps, cfg)

    model = model.to(device)

    best_valid_loss = torch.inf

    for epoch in trange(cfg['n_epochs']):
        with noise_hook, optimizer_hook:
            model.train()
            train_one_epoch(training_loader, optimizer, noise_scheduler, model, loss_fn, cfg['num_noise_samples_batch'], cfg['num_noise_samples_accumulation'], device)

        model.eval()
        avg_train_loss, avg_valid_loss, avg_test_loss = compute_01_loss(model, training_loader, validation_loader, test_loader, device)
        if noise_scheduler is not None:
            noise_scheduler.update_from_losses(avg_train_loss.item(), avg_valid_loss.item())
            writer.add_scalar('scheduler/scale', noise_scheduler.get_noise_scalar(), epoch)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss

            atomic_write_json(best_loss_path, {
                "epoch_0_based": epoch,
                "train_01": avg_train_loss.item(),
                "valid_01": avg_valid_loss.item(),
                "test_01": avg_test_loss.item(),
                "cfg": cfg,
            })

        writer.add_scalar('loss_01/train', avg_train_loss.item(), epoch)
        writer.add_scalar('loss_01/valid', avg_valid_loss.item(), epoch)
        writer.add_scalar('loss_01/test', avg_test_loss.item(), epoch)

    if noise_scheduler is not None:
        noise_scheduler.update_from_run(best_valid_loss.item())

    writer.close()
    print("\nTraining finished.")


def get_optimizer_hook(optimizer, cfg):
    if cfg.get("covariance_mode") in ("bineta",):
        return GradsStatsHook(optimizer, alpha=0.01, sigma_min=1e-3, history_size=5, gamma=0.1)
    else:
        return contextlib.nullcontext()


def get_noise_hook(model, optimizer, total_steps, cfg):
    noise_hook = contextlib.nullcontext()
    noise_scheduler = get_noise_scheduler(total_steps, cfg)

    if noise_scheduler is not None:
        noise_scheduler.amend_config(cfg)

    if cfg['noise_std'] is not None:
        var_provider = get_var_provider(optimizer, cfg)
        noise_hook = NoiseHook(model, var_provider, noise_scheduler)

    return noise_hook, noise_scheduler


def get_var_provider(optimizer, cfg):
    covariance_mode = cfg['covariance_mode']
    if covariance_mode not in VARIANCE_PROVIDER_FACTORIES:
        raise RuntimeError(
            f"Invalid covariance mode '{covariance_mode}'. Options are: {list(VARIANCE_PROVIDER_FACTORIES)}"
        )
    return VARIANCE_PROVIDER_FACTORIES[covariance_mode](optimizer, cfg)


def get_noise_scheduler(total_steps, cfg):
    noise_scheduler_id = cfg['noise_scheduler']
    if noise_scheduler_id not in NOISE_SCHEDULER_FACTORIES:
        raise RuntimeError(f"Invalid noise scheduler '{noise_scheduler_id}'. Options are: {list(NOISE_SCHEDULER_FACTORIES)}")
    return NOISE_SCHEDULER_FACTORIES[noise_scheduler_id](total_steps, cfg)


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
    tmp.replace(path)


def to_jsonable(cfg):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out[k] = to_jsonable(v)
        elif isinstance(v, pathlib.Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


if __name__ == '__main__':
    ablate_train_valid_diff_pid()
