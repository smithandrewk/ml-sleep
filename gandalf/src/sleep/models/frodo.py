import torch
from torch import nn

from sleep.models.residual import ResidualBlock


class Frodo(nn.Module):
    def __init__(self, n_features=5000):
        super().__init__()
        self.n_features = n_features
        self.block1 = ResidualBlock(1, 8, n_features)
        self.block2 = ResidualBlock(8, 16, n_features)
        self.block3 = ResidualBlock(16, 16, n_features)

        self.gap = nn.AvgPool1d(kernel_size=n_features)
        self.fc1 = nn.Linear(in_features=16, out_features=3)

    def forward(self, x, classification=True):
        x = x.view(-1, 1, self.n_features)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        if classification:
            x = self.fc1(x.squeeze())
            return x
        else:
            return x.squeeze()
