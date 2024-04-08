import os
import re
from importlib import resources
from pathlib import PosixPath
from typing import Union

import torch
from torch.utils import data

dataset_path = resources.files("dexskills") / "DexSkill_dataset" / "dataset"
LHdataset_path = resources.files("dexskills") / "DexSkill_dataset" / "LH_dataset"


def extract_number(f):
    s = re.findall(r"\d+", f)
    return (int(s[0]) if s else -1, f)


class DexSkillsDataset(data.Dataset):
    def __init__(self, data_dir: Union[str, PosixPath] = dataset_path):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        # Load the preprocessed data
        data = torch.load(file_path)
        return data


class LongHorizonTasksDataset(data.Dataset):
    def __init__(self, data_dir: Union[str, PosixPath] = LHdataset_path):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(data_dir), key=extract_number)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        # Load the preprocessed data
        data = torch.load(file_path)
        return data
