import torch.nn as nn
import torch.nn.functional as F


class GarmentClassifier(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.fc3(x)
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
