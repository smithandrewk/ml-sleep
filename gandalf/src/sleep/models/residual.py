import torch
from torch import nn
from torch.nn.functional import relu


class ResidualBlock(nn.Module):
    def __init__(self, in_feature_maps, out_feature_maps, n_features):
        super().__init__()
        self.c1 = nn.Conv1d(in_feature_maps, out_feature_maps, kernel_size=8, padding='same', bias=False)
        self.bn1 = nn.LayerNorm((out_feature_maps, n_features), elementwise_affine=False)

        self.c2 = nn.Conv1d(out_feature_maps, out_feature_maps, kernel_size=5, padding='same', bias=False)
        self.bn2 = nn.LayerNorm((out_feature_maps, n_features), elementwise_affine=False)

        self.c3 = nn.Conv1d(out_feature_maps, out_feature_maps, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.LayerNorm((out_feature_maps, n_features), elementwise_affine=False)

        self.c4 = nn.Conv1d(in_feature_maps, out_feature_maps, 1, padding='same', bias=False)
        self.bn4 = nn.LayerNorm((out_feature_maps, n_features), elementwise_affine=False)

    def forward(self, x):
        identity = x
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = relu(x)

        identity = self.c4(identity)
        identity = self.bn4(identity)

        x = x + identity
        x = relu(x)

        return x
