import torch


class ZeroOneLoss(torch.nn.Module):
    def __init__(self):
        super(ZeroOneLoss, self).__init__()

    def forward(self, input, target):
        input_rounded = torch.argmax(input, 1)
        return torch.mean((input_rounded != target).to(torch.float))


@torch.no_grad()
def compute_01_loss(model, training_loader, validation_loader):
    avg_tloss = 0.0
    avg_vloss = 0.0
    zo_loss = ZeroOneLoss()

    for i, (tinputs, tlabels) in enumerate(training_loader, start=1):
        if torch.cuda.is_available():
            tinputs = tinputs.cuda()
            tlabels = tlabels.cuda()
        toutputs = model(tinputs)
        tloss = zo_loss(toutputs, tlabels)

        ratio = 1 / i
        avg_tloss = (1-ratio)*avg_tloss + ratio*tloss

    for i, (vinputs, vlabels) in enumerate(validation_loader, start=1):
        # If GPU is available: make use of it
        if torch.cuda.is_available():
            vinputs = vinputs.cuda()
            vlabels = vlabels.cuda()
        voutputs = model(vinputs)
        vloss = zo_loss(voutputs, vlabels)

        ratio = 1 / i
        avg_vloss = (1-ratio)*avg_vloss + ratio*vloss

    return avg_tloss, avg_vloss
