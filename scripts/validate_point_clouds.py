import os
import plyfile
from tqdm import tqdm
import numpy as np
import json

point_clouds = os.listdir('./resources/dataset/')
categories = json.load(open('./resources/categories.json'))
checklist = dict.fromkeys(categories.keys(), 0)
checklist['0'] = 0

# find value in a dictionary of lists
def find_category(dictionary, value) -> str:
    for key, values in dictionary.items():
        if value in values:
            return key
    return '0'

for pc in tqdm(point_clouds, desc='Processing point clouds'):
    ply_path = f'./resources/dataset/{pc}'
    ply = plyfile.PlyData.read(ply_path)
    num_verts = ply['vertex'].count
    vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
    vertices[:, 0] = ply['vertex'].data['x']
    vertices[:, 1] = ply['vertex'].data['y']
    vertices[:, 2] = ply['vertex'].data['z']
    vertices[:, 3] = np.interp(ply['vertex'].data['red'], [0, 255], [0, 1])
    vertices[:, 4] = np.interp(ply['vertex'].data['blue'], [0, 255], [0, 1])
    vertices[:, 5] = np.interp(ply['vertex'].data['green'], [0, 255], [0, 1])

    if vertices.shape != (16384, 6):
        print(f'{pc} has wrong shape. Expected: (16384, 6), got: {vertices.shape}')

    cat = find_category(categories, pc.split('.')[0])
    checklist[cat] += 1

print(f'Couldnt find {checklist["0"]} point clouds')
for cat in tqdm(categories, desc='Checking categories'):
    if checklist[cat] != len(cat):
        print(f'{cat} has {checklist[cat]} instead of {len(cat)} point clouds')
