import torch
from torch import nn

from sleep.models.frodo import Frodo


class Gandalf(nn.Module):
    def __init__(self, n_features=5000, lstm_hidden=32, n_classes=3, sequence_length=9):
        super().__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoder = Frodo(n_features=n_features)
        self.lstm = nn.LSTM(16, lstm_hidden, bidirectional=True)
        self.fc1 = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x, classification=True):
        x = x.view(-1, self.sequence_length, self.n_features)
        batch = x.size(0)
        x = x.view(batch * self.sequence_length, 1, self.n_features)
        encoded = self.encoder(x, classification=False)
        encoded = encoded.view(self.sequence_length, batch, -1)
        out, _ = self.lstm(encoded)
        if classification:
            return self.fc1(out[-1])
        else:
            return out[-1]
