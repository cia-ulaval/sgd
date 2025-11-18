from time import time
from torch.utils.tensorboard import SummaryWriter
from src.utils import create_run_name
from src.data import dataset_creator
from src.loss_function import compute_01_loss
from src.model import *
from src.seed import set_seed
from src.training import train_one_epoch
from src.variance_provider import ConstantVarianceProvider, AdamSqGradsVarianceProvider, InvAdamSqGradsVarianceProvider
from tqdm import trange


VARIANCE_PROVIDER_FACTORIES = {
    "inv_sq_grads": lambda optimizer, cfg: InvAdamSqGradsVarianceProvider(optimizer, 0.0, cfg['noise_std'] ** 2),
    "sq_grads": lambda optimizer, cfg: AdamSqGradsVarianceProvider(optimizer, 0.0, cfg['noise_std'] ** 2),
    "isotropic": lambda optimizer, cfg: ConstantVarianceProvider(cfg['noise_std']**2),
}


def main(cfg):
    set_seed(cfg['seed'])
    writer = SummaryWriter(f"./logs/{create_run_name(cfg)}")

    print("\nStandard training loop initialized.\n")

    # We initialize the various component for our training.
    training_loader, validation_loader, classes = dataset_creator()
    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    covariance_mode = cfg['covariance_mode']
    if covariance_mode not in VARIANCE_PROVIDER_FACTORIES:
        raise RuntimeError(f"Invalid covariance mode '{covariance_mode}'. Options are: {list(VARIANCE_PROVIDER_FACTORIES)}")
    var_provider = VARIANCE_PROVIDER_FACTORIES[covariance_mode](optimizer, cfg)
    model.set_variance_provider(var_provider)

    # If GPU is available: make use of it
    if torch.cuda.is_available():
        model = model.cuda()

    epoch_number = 0

    best_vloss = torch.inf

    t_init = time()
    for epoch in trange(cfg['n_epochs']):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        train_one_epoch(training_loader, optimizer, model, loss_fn)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        avg_tloss, avg_vloss = compute_01_loss(model, training_loader, validation_loader)
        cur_time = round(time() - t_init, 2)
        print('Epoch number: {}; Train 0-1 loss: {}; Valid 0-1 loss: {}; Running time: {}s.'.format(
            epoch_number + 1, round(avg_tloss.item(), 4), round(avg_vloss.item(), 4), cur_time))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        writer.add_scalar('loss/train', avg_tloss.item(), epoch)
        writer.add_scalar('loss/valid', avg_vloss.item(), epoch)

        epoch_number += 1

    writer.close()
    print("\nTraining finished.")


if __name__ == '__main__':
    config = {
        'project_name': 'SSGD',
        'seed': 20250729,
        'dataset': 'CIFAR100',
        'noise_std': 0,
        'covariance_mode': "isotropic",  # 'isotropic', 'sq_grads', 'inv_sq_grads'
        'n_epochs': 40,
        'lr': 1e-3,
    }

    main(config)
