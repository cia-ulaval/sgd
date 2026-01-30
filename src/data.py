import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def dataset_creator(
    val_fraction=0.1,
    batch_size=512,
    num_workers=4,
    seed=42,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Create datasets for training & validation, download if necessary
    full_train_set = torchvision.datasets.CIFAR100('./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR100('./data', train=False, transform=transform, download=True)

    n_total = len(full_train_set)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train_set, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    classes = full_train_set.classes
    return train_loader, val_loader, test_loader, classes
