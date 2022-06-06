import os
import random
from typing import List

import torch
import torch.utils.data as data
import numpy as np
import plyfile
import json

class ShapeNet(data.Dataset):
    
    def __init__(self, data_path, split) -> None:
        assert split in ["train", "test_seen", "test_unseen", "validate"], "split error value!"

        self.data_path = data_path
        self.split = split
        self.paths = self.load_data()

        self.seen_cat = [
            "02691156",  # airplane
            "02933112",  # cabinet
            "02958343",  # car
            "03001627",  # chair
            "03636649",  # lamp
            "04256520",  # sofa
            "04379243",  # table
            "04530566",  # boat
        ]

        self.unseen_cat = [
            "02924116",  # bus
            "02818832",  # bed
            "02871439",  # bookshelf
            "02828884",  # bench
            "03467517",  # guitar
            "03790512",  # motorbike
            "04225987",  # skateboard
            "03948459",  # pistol
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        path = self.paths[index]
        vertices = self.read_point_cloud(f'{self.data_path}/{path}.ply')

        return torch.from_numpy(vertices)
        
    def load_data(self) -> List[str]:
        categories = json.load(open('./resources/categories.json'))
        paths: List[str] = []
        
        if self.split == "test_unseen":
            for key in self.seen_cat:
                categories.pop(key)
        else:
            for key in self.unseen_cat:
                categories.pop(key)

        if self.split == "test_unseen" or self.split == "test_seen":
            self.split = "test"

        for cat in categories:
            for model in cat:
                if model[1] == self.split:
                    paths.append(model[0])

        return paths

    def read_point_cloud(self, path):
        ply = plyfile.PlyData.read(path)
        num_verts = ply['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = ply['vertex'].data['x']
        vertices[:, 1] = ply['vertex'].data['y']
        vertices[:, 2] = ply['vertex'].data['z']
        vertices[:, 3] = np.interp(ply['vertex'].data['red'], [0, 255], [0, 1])
        vertices[:, 4] = np.interp(ply['vertex'].data['blue'], [0, 255], [0, 1])
        vertices[:, 5] = np.interp(ply['vertex'].data['green'], [0, 255], [0, 1])

        return vertices

