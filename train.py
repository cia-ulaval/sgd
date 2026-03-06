import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from laplace_model import FlexibleGarmentClassifier
from loss_fct import compute_01_loss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    valset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    training_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validation_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return training_loader, validation_loader


def train_one_epoch(model, training_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in training_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(training_loader)


def evaluate_ce(model, validation_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(validation_loader)


def train_model_for_noise_mode(
    noise_mode,
    noise_std,
    num_epochs,
    learning_rate,
    momentum,
    weight_decay,
    training_loader,
    validation_loader,
    device
):

    print(f"\n--- Training mode: {noise_mode} ---")

    model = FlexibleGarmentClassifier(
        noise_mode=noise_mode,
        noise_std=noise_std
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    val_ce_losses = []

    for epoch in range(num_epochs):

        train_ce = train_one_epoch(
            model,
            training_loader,
            criterion,
            optimizer,
            device
        )

        val_ce = evaluate_ce(
            model,
            validation_loader,
            criterion,
            device
        )

        train_01, val_01 = compute_01_loss(
            model,
            training_loader,
            validation_loader
        )

        val_ce_losses.append(val_ce)

        print(
            f"Mode {noise_mode} | "
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train CE: {train_ce:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Train 0-1: {train_01:.4f} | "
            f"Val 0-1: {val_01:.4f}"
        )

    return val_ce_losses


def plot_all_val_ce(results):
    epochs = range(1, len(next(iter(results.values()))) + 1)

    plt.figure(figsize=(10, 6))

    for noise_mode, losses in results.items():
        plt.plot(epochs, losses, label=f"{noise_mode} - Val CE")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross-Entropy Loss")
    plt.title("Validation CE Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():

    batch_size = 64
    num_epochs = 40
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0
    noise_std = 0.05

    noise_modes = ["none", "prop", "laplace"]

    device = get_device()

    training_loader, validation_loader = get_data_loaders(batch_size)

    results = {}

    for noise_mode in noise_modes:

        current_noise_std = 0.0 if noise_mode == "none" else noise_std

        results[noise_mode] = train_model_for_noise_mode(
            noise_mode,
            current_noise_std,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
            training_loader,
            validation_loader,
            device
        )

    plot_all_val_ce(results)


if __name__ == "__main__":
    main()
