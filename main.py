from time import time
import wandb
wandb.login()

from src.utils import create_run_name
from src.data import dataset_creator
from src.loss_function import compute_01_loss
from src.model import *
from src.seed import set_seed
from src.training import train_one_epoch

def main(cfg):
    set_seed(cfg['seed'])
    run_name = create_run_name(cfg)
    if cfg['is_using_wandb']:
        wandb.init(name=str(run_name), project=cfg['project_name'], config=cfg)

    print("\nStandard training loop initialized.\n")

    # We initialize the various component for our training.
    training_loader, validation_loader, classes = dataset_creator()
    model = GarmentClassifier(noise_type=cfg['noise_type'], noise_std=cfg['noise_std'])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # If GPU is available: make use of it
    if torch.cuda.is_available():
        model = model.cuda()

    # Initializing in a separate cell so we can easily add more epochs to the same run
    epoch_number = 0

    best_vloss = torch.inf

    t_init = time()
    for epoch in range(cfg['n_epochs']):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
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

        seed_results = {'train_loss': avg_tloss, 'valid_loss': avg_vloss}
        if cfg['is_using_wandb']:
            wandb.log(seed_results)

        epoch_number += 1
    if cfg['is_using_wandb']:
        wandb.finish()
    print("\nTraining finished.")

if __name__=='__main__':
    config = {'project_name': 'SSGD',
              'is_using_wandb': True,
              'seed': 20250729,
              'dataset': 'CIFAR100',
              'noise_std': None,
              'noise_type': 'prop',
              'n_epochs': 100,
              'lr': 0.01}
    for noise_std in [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0]:
        config['noise_std'] = noise_std
        main(config)