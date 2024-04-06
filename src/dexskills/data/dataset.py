import os

import torch
from torch.utils import data


class DexSkillsDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        # Load the preprocessed data
        data = torch.load(file_path)
        return data
