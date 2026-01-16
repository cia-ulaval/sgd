import torch


class ZeroOneLoss(torch.nn.Module):
    def __init__(self):
        super(ZeroOneLoss, self).__init__()

    def forward(self, input, target):
        input_rounded = torch.argmax(input, 1)
        return torch.mean((input_rounded != target).to(torch.float))


@torch.no_grad()
def compute_01_loss(model, training_loader, validation_loader, test_loader):
    avg_train_loss = 0.0
    avg_valid_loss = 0.0
    avg_test_loss = 0.0
    zo_loss = ZeroOneLoss()

    for i, (inputs, labels) in enumerate(training_loader, start=1):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = zo_loss(outputs, labels)

        ratio = 1 / i
        avg_train_loss = (1-ratio)*avg_train_loss + ratio*loss

    for i, (inputs, labels) in enumerate(validation_loader, start=1):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = zo_loss(outputs, labels)

        ratio = 1 / i
        avg_valid_loss = (1-ratio)*avg_valid_loss + ratio*loss

    for i, (inputs, labels) in enumerate(test_loader, start=1):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = zo_loss(outputs, labels)

        ratio = 1 / i
        avg_test_loss = (1-ratio)*avg_test_loss + ratio*loss

    return avg_train_loss, avg_valid_loss, avg_test_loss
