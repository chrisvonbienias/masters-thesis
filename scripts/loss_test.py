import torch
from time import time
from utils.model_utils import calc_dcd, calc_emd, calc_cd
from tqdm import tqdm
import numpy as np
import plyfile


def read_point_cloud(path):
        ply = plyfile.PlyData.read(path)
        num_verts = ply['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = ply['vertex'].data['x']
        vertices[:, 1] = ply['vertex'].data['y']
        vertices[:, 2] = ply['vertex'].data['z']
        vertices[:, 3] = np.interp(ply['vertex'].data['red'], [0, 255], [0, 1])
        vertices[:, 4] = np.interp(ply['vertex'].data['green'], [0, 255], [0, 1])
        vertices[:, 5] = np.interp(ply['vertex'].data['blue'], [0, 255], [0, 1])

        return vertices
    

dcd_list = []
emd_list = []
cd_list = []
dcd_val = []
emd_val = []
cd_val = []

for i in tqdm(range(100)):
    points1 = torch.rand(8, 16384, 6).cuda()
    points2 = torch.rand(8, 16384, 6, requires_grad=True).cuda()
    
    clock = time()
    dcd, _, _ = calc_dcd(points1, points2)
    dcd_val.append(torch.mean(dcd))
    dcd_list.append(time() - clock)
    
    clock = time()
    emd = calc_emd(points1, points2)
    emd_val.append(torch.mean(emd))
    emd_list.append(time() - clock)
    
    clock = time()
    _, _, cd = calc_cd(points1, points2, calc_f1=True)
    cd_val.append(torch.mean(cd))
    cd_list.append(time() - clock)

print(f'DCD finshed in: {sum(dcd) / len(dcd)}. Avg. value = {sum(dcd_val) / len(dcd_val)}')
print(f'EMD finshed in: {sum(emd) / len(emd)}. Avg. value = {sum(emd_val) / len(emd_val)}')
print(f'CD finshed in: {sum(cd) / len(cd)}. Avg. value = {sum(cd_val) / len(cd_val)}')


pt1 = read_point_cloud('resources/dataset/complete/1a0a2715462499fbf9029695a3277412.ply')
pt2 = read_point_cloud('resources/dataset/complete/1a00aa6b75362cc5b324368d54a7416f.ply')
pt11 = []
pt22 = []

for i in range(16384):
    pt11.append(pt1[i][:3])
    pt22.append(pt2[i][:3])
    
pt11 = torch.Tensor(pt11).cuda()
pt22 = torch.Tensor(pt22).cuda()
pt11 = torch.unsqueeze(pt11, dim=0)
pt22 = torch.unsqueeze(pt22, dim=0)
pt1 = torch.from_numpy(pt1).cuda()
pt2 = torch.from_numpy(pt2).cuda()
pt1 = torch.unsqueeze(pt1, dim=0)
pt2 = torch.unsqueeze(pt2, dim=0)

print(pt1.shape)
print(pt11.shape)

dcd1, _, _ = calc_dcd(pt1, pt2)
dcd2, _, _ = calc_dcd(pt11, pt22)

print(dcd1)
print(dcd2)