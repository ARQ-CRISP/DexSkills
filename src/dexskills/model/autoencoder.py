import lightning as L
import torch
from dexskills.model.blocks import BatchedBlock, FCBlock, ResidualBlock
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        bottleneck_size: int,
        hidden_size1: int = 1024,
        hidden_size2: int = 512,
        leaky_relu_negative_slope=0.01,
    ):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            BatchedBlock(input_size, hidden_size1, leaky_relu_negative_slope),
            BatchedBlock(hidden_size1, hidden_size2, leaky_relu_negative_slope),
            BatchedBlock(hidden_size2, hidden_size2, leaky_relu_negative_slope),
            BatchedBlock(hidden_size2, hidden_size2, leaky_relu_negative_slope),
            FCBlock(hidden_size2, hidden_size2),
            FCBlock(hidden_size2, bottleneck_size),
            ResidualBlock(bottleneck_size),
            # Additional layers to reach 16 linear layers
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        bottleneck_size: int,
        hidden_size1: int = 512,
        hidden_size2: int = 1024,
        leaky_relu_negative_slope=0.01,
    ):
        super(Decoder, self).__init__()

        self.network = nn.Sequential(
            ResidualBlock(bottleneck_size),
            FCBlock(bottleneck_size, hidden_size1),
            FCBlock(hidden_size1, hidden_size1),
            BatchedBlock(hidden_size1, hidden_size1, leaky_relu_negative_slope),
            BatchedBlock(hidden_size1, hidden_size1, leaky_relu_negative_slope),
            BatchedBlock(hidden_size1, hidden_size1, leaky_relu_negative_slope),
            BatchedBlock(hidden_size1, hidden_size2, leaky_relu_negative_slope),
            nn.Linear(hidden_size2, input_size),
            # ... (Continue adding layers until you have 16 linear layers in total)
        )

    def forward(self, x):
        return self.network(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_size=143,
        bottleneck_size=64,
        hidden_size1: int = 1024,
        hidden_size2: int = 512,
    ):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size

        self.encoder = Encoder(input_size, bottleneck_size, hidden_size1, hidden_size2)
        self.decoder = Decoder(input_size, bottleneck_size, hidden_size2, hidden_size1)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(x)
        return out


# class AutoEncoder(L.LightningModule):
#     def __init__(
#         self,
#         input_size=143,
#         bottleneck_size=64,
#         hidden_size1: int = 1024,
#         hidden_size2: int = 512,
#     ):
#         super().__init__()
#         self.input_size = input_size

#         self.encoder = Encoder(input_size, bottleneck_size, hidden_size1, hidden_size2)
#         self.decoder = Decoder(input_size, bottleneck_size, hidden_size2, hidden_size1)

#     def forward(self, x):
#         out = self.encoder(x)
#         out = self.decoder(x)
#         return out

#     def configure_optimizers(self):
#         en_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
#         de_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
#         return [en_optimizer, de_optimizer]
