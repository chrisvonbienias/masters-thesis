import os
from typing import Any, List

import torch
import torch.utils.data as data
import numpy as np
import plyfile
import json
from skimage.color import rgb2lab

class ShapeNet(data.Dataset):
    
    def __init__(self, data_path, split, num_dense=8192, subset=None) -> None:
        assert split in ["train", "test_seen", "test_unseen", "validate"], "split error value!"
        
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
        
        self.data_path = data_path
        self.split = split
        self.num_dense = num_dense
        self.subset = subset
        self.paths = self.load_data()
        

    def __len__(self) -> int:
        return len(self.paths)
    

    def __getitem__(self, index) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        path = self.paths[index]
        complete_path = f'{self.data_path}/complete/{path}.ply'
        partial_path = f'{self.data_path}/partial/{path}.ply'
        complete = self.random_sample(self.read_point_cloud(complete_path), self.num_dense)
        partial = self.random_sample(self.read_point_cloud(partial_path), self.num_dense // 4)

        return torch.from_numpy(complete), torch.from_numpy(partial)
    
        
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

        for cat in categories.values():
            for model in cat:
                if not os.path.exists(f'resources/dataset/complete/{model[0]}.ply'):
                    continue
                if not os.path.exists(f'resources/dataset/partial/{model[0]}.ply'):
                    continue

                if model[1] == self.split:
                    paths.append(model[0])
                    
        if self.subset is not None:
            paths = self.random_sample(np.asarray(paths), len(paths) // self.subset)

        return list(paths)


    def read_point_cloud(self, path):
        ply = plyfile.PlyData.read(path)
        num_verts = ply['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = ply['vertex'].data['x']
        vertices[:, 1] = ply['vertex'].data['y']
        vertices[:, 2] = ply['vertex'].data['z']
        vertices[:, 3] = ply['vertex'].data['red']
        vertices[:, 4] = ply['vertex'].data['green']
        vertices[:, 5] = ply['vertex'].data['blue']
        vertices[:, 3:] = rgb2lab(vertices[:, 3:] / 255, illuminant='D65') / 100

        return vertices

    def random_sample(self, pc, n) -> Any:
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]