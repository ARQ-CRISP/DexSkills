import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_features):
        super(ResidualBlock, self).__init__()
        
        self.fc = nn.Linear(input_features, input_features)
        self.bn = nn.BatchNorm1d(input_features)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):

        out = self.fc(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out + x


class Autoencoder(nn.Module):
    def __init__(self, input_size=143, bottleneck_size=64):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
 
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, bottleneck_size),  # Additional layers to reach 16 linear layers
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(bottleneck_size)
        self.res2 = ResidualBlock(bottleneck_size)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(1024, input_size)
            # ... (Continue adding layers until you have 16 linear layers in total)
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        x = self.encoder(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.decoder(x)
        # x = x.view(-1, 1, self.input_size)  # Reshape back to original input shape
        return x
    def get_latent_vector(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.res1(x)
        return x


class CustomNetwork(nn.Module):
    def __init__(self, num_classes, key_frame_output_size = 8, input_size=20):
        super(CustomNetwork, self).__init__()


        # Fully Connected Layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output Layers
        self.classifier_output = nn.Linear(128, num_classes)
        self.contact_indicator_output = nn.Linear(128, 1)
        self.key_frame_output = nn.Linear(128, key_frame_output_size)

    def forward(self, x):

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output Layers
        # classifier_out = F.softmax(self.classifier_output(x), dim=1)
        classifier_out = self.classifier_output(x)
        # contact_indicator_out = torch.sigmoid(self.contact_indicator_output(x))
        # key_frame_out = self.key_frame_output(x)  # Apply softmax/relu/linear based on your requirement

        return classifier_out

