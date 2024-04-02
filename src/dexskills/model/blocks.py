from typing import Dict, Optional, Type

import torch.nn.functional as F
from torch import nn


class FCBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: Type[nn.Module] = nn.ReLU,
        activation_kwargs: Optional[Dict] = None,
    ):
        super(FCBlock, self).__init__()
        activation_kwargs = activation_kwargs or {}
        self.fc = nn.Linear(input_size, hidden_size)
        self.activation = activation(**activation_kwargs)

    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out


class BatchedBlock(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, leaky_relu_negative_slope: float = 1e-2
    ):
        super(BatchedBlock, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

    def forward(self, x):

        out = self.fc(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out


class ResidualBlock(BatchedBlock):
    def __init__(self, input_features: int, leaky_relu_negative_slope: float = 1e-2):
        super(ResidualBlock, self).__init__(
            input_features, input_features, leaky_relu_negative_slope
        )

        # self.fc = nn.Linear(input_features, input_features)
        # self.bn = nn.BatchNorm1d(input_features)
        # self.lrelu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

    def forward(self, x):
        out = super(ResidualBlock, self).forward(x)

        return out + x
