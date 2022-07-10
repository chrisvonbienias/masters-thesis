import torch
from time import time
from utils.model_utils import calc_dcd, calc_emd, calc_cd

points1 = torch.rand(1, 8192, 6).cuda()
points2 = torch.rand(1, 8192, 6, requires_grad=True).cuda()

clock = time()
dcd, _, _ = calc_dcd(points1, points2)
print(f'DCD finshed in: {time() - clock}. DCD = {dcd[0]}')

clock = time()
emd = calc_emd(points1, points2)
print(f'EMD finshed in: {time() - clock}. EMD = {emd[0]}')

clock = time()
cd, _ = calc_cd(points1, points2)
print(f'CD finshed in: {time() - clock}. CD = {cd[0]}')