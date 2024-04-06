import numpy as np
import torch
from data.constants import label_names
from data.dataset import DexSkillsDataset
from model.autoencoder import AutoEncoder
from model.classifier import ClassifierNet
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

autoencoder = AutoEncoder(
    (144 - 70 + 3 + 3 + 3) * 3 + (23 + 4) * 3,
)

classifier = ClassifierNet()
