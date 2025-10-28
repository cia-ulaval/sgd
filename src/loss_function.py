import torch

class ZeroOneLoss(torch.nn.Module):
    def __init__(self):
        super(ZeroOneLoss, self).__init__()

    def forward(self, input, target):
        input_rounded = torch.argmax(input, 1)
        return torch.mean((input_rounded != target).to(torch.float))

def compute_01_loss(model, training_loader, validation_loader):
    running_tloss = 0.0
    running_vloss = 0.0
    zo_loss = ZeroOneLoss()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, tdata in enumerate(training_loader):
            tinputs, tlabels = tdata
            # If GPU is available: make use of it
            if torch.cuda.is_available():
                tinputs = tinputs.cuda()
                tlabels = tlabels.cuda()
            toutputs = model(tinputs)
            tloss = zo_loss(toutputs, tlabels)
            running_tloss += tloss
        avg_tloss = running_tloss / (i + 1)

        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            # If GPU is available: make use of it
            if torch.cuda.is_available():
                vinputs = vinputs.cuda()
                vlabels = vlabels.cuda()
            voutputs = model(vinputs)
            vloss = zo_loss(voutputs, vlabels)
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)

    return avg_tloss, avg_vloss