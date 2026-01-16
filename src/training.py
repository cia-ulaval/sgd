import torch


def train_one_epoch(training_loader, optimizer, noise_scheduler, model, loss_fn, num_noise_samples_batch=1, num_noise_samples_accumulation=1):
    for inputs, labels in training_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        inputs = inputs.repeat([num_noise_samples_batch] + [1] * (len(inputs.shape) - 1))
        labels = labels.repeat([num_noise_samples_batch] + [1] * (len(labels.shape) - 1))

        for _ in range(max(1, num_noise_samples_accumulation)):
            outputs = model(inputs)

            loss = loss_fn(outputs, labels) / max(1, num_noise_samples_accumulation)
            loss.backward()

        optimizer.step()
        if noise_scheduler is not None:
            noise_scheduler.step()
