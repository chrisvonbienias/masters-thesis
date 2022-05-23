import os
import random

import torch
import torch.utils.data as data
import numpy as np

class ShapeNet(data.Dataset):
    
    def __init__(self, data_path, split) -> None:
        assert split in ["train", "test", "valid"], "split error value!"

        self.data_path = data_path
        self.split = split
        self.paths = self.load_data()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def load_data(self):
        pass

