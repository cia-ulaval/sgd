import torch


def train_one_epoch(training_loader, optimizer, noise_scheduler, model, loss_fn, num_noise_samples=1):
    for inputs, labels in training_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        for sample_i in range(1, 1 + max(1, num_noise_samples)):
            outputs = model(inputs)

            loss = loss_fn(outputs, labels) / num_noise_samples
            loss.backward()

        optimizer.step()
        if noise_scheduler is not None:
            noise_scheduler.step()
