from typing import List, Optional

from dexskills.model.blocks import FCBlock
from torch import nn


class ClassifierNet(nn.Module):
    def __init__(
        self, input_size: int, num_classes: int, hidden_dims: Optional[List[int]] = None
    ) -> None:
        super(ClassifierNet, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        hidden_dims = hidden_dims or [512, 256, 128]
        net_dims = [input_size] + hidden_dims

        self.network = nn.Sequential(
            *[
                FCBlock(in_dim, out_dim)
                for in_dim, out_dim in zip(net_dims[:-1], net_dims[1:])
            ],
            nn.Linear(net_dims[-1], self.num_classes)
        )

    def forward(self, x):
        return self.network(x)
