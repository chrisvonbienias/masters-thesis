# Functions for parsing and splitting the dataset

import os
import plyfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from shutil import copyfile


def split_dataset() -> None:

    models_dir = "./resources/3d_models/"
    train_dir = "./resources/dataset/train/"
    test_dir = "./resources/dataset/test/"
    models = os.listdir(models_dir)
    # keep only .ply files
    models = [x for x in models if x[-1] == 'y']

    train_set, test_set = train_test_split(models, train_size=0.7)

    for data in tqdm(train_set, desc="Moving training dataset"):
        src = models_dir + data
        dst = train_dir + data
        copyfile(src, dst)A

    for data in tqdm(test_set, desc="Moving testing dataset"):
        src = models_dir + data
        dst = test_dir + data
        copyfile(src, dst)

    print("Files moved successfully!")


def parse_dataset():

    train_points = []
    test_points = []

    dataset_dir = './resources/dataset/'
    dataset = os.listdir(dataset_dir)
    test_dataset = os.listdir(dataset_dir + dataset[0])
    train_dataset = os.listdir(dataset_dir + dataset[1])

    # Testing dataset
    for data in tqdm(test_dataset, desc="Processing testing dataset"):

        ply_path = dataset_dir + 'test/' + data
        ply = plyfile.PlyData.read(ply_path)
        num_verts = ply['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = ply['vertex'].data['x']
        vertices[:, 1] = ply['vertex'].data['y']
        vertices[:, 2] = ply['vertex'].data['z']
        vertices[:, 3] = np.interp(ply['vertex'].data['red'], [0, 255], [0, 1])
        vertices[:, 4] = np.interp(ply['vertex'].data['blue'], [0, 255], [0, 1])
        vertices[:, 5] = np.interp(ply['vertex'].data['green'], [0, 255], [0, 1])

        test_points.append(vertices)

    # Training dataset
    for data in tqdm(train_dataset, desc="Processing training dataset"):
        ply_path = dataset_dir + 'train/' + data
        ply = plyfile.PlyData.read(ply_path)
        num_verts = ply['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = ply['vertex'].data['x']
        vertices[:, 1] = ply['vertex'].data['y']
        vertices[:, 2] = ply['vertex'].data['z']
        vertices[:, 3] = np.interp(ply['vertex'].data['red'], [0, 255], [0, 1])
        vertices[:, 4] = np.interp(ply['vertex'].data['blue'], [0, 255], [0, 1])
        vertices[:, 5] = np.interp(ply['vertex'].data['green'], [0, 255], [0, 1])

        train_points.append(vertices)

    return np.array(train_points), np.array(test_points)
