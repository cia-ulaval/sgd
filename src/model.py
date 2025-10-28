import torch
import torch.nn as nn
import torch.nn.functional as F

def noise(x, noise_type, noise_std, length):
    if noise_std == 0:
        return 0

    norms = torch.sum(x ** 2, dim=1)
    norms_augm = noise_std * (norms + 1)
    norms_augm_dupl = norms_augm.repeat_interleave(length)
    if noise_type == 'prop':
        white_noise = torch.normal(0, norms_augm_dupl)
    if torch.cuda.is_available():
        white_noise = white_noise.cuda()
    return white_noise.reshape(-1, length)

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self, noise_type, noise_std=0):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm1d(num_features=400)
        self.fc1 = nn.Linear(400, 120)
        self.bn2 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(120, 120)
        self.bn3 = nn.BatchNorm1d(num_features=120)
        self.fc3 = nn.Linear(120, 100)
        self.noise_type = noise_type
        self.noise_std = noise_std

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = self.bn1(x)
        x = F.relu(self.fc1(x)) + noise(x, self.noise_type, self.noise_std, 120)
        x = self.bn2(x)
        x = F.relu(self.fc2(x)) + noise(x, self.noise_type, self.noise_std, 120)
        x = self.bn3(x)
        x = self.fc3(x) + noise(x, self.noise_type, self.noise_std, 100)
        return x


class SimpleMLPClassifier(nn.Module):
    def __init__(self):
        super(SimpleMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x