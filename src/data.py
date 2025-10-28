import torch
import torchvision
import torchvision.transforms as transforms


def dataset_creator():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.CIFAR100('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.CIFAR100('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=512, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=512, shuffle=False)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    return training_loader, validation_loader, classes