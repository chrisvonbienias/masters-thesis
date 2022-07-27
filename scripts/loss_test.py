import torch
from time import time
from utils.model_utils import calc_dcd, calc_emd, calc_cd
from tqdm import tqdm
import numpy as np
    
dcd_list = []
emd_list = []
cd_list = []
dcd_val = []
emd_val = []
cd_val = []

for i in tqdm(range(100)):
    points1 = torch.rand(32, 8192, 6).cuda()
    points2 = torch.rand(32, 8192, 6, requires_grad=True).cuda()
    
    clock = time()
    dcd, _, _ = calc_dcd(points1, points2)
    dcd_val.append(torch.mean(dcd))
    dcd_list.append(time() - clock)
    
    clock = time()
    emd = calc_emd(points1, points2)
    emd_val.append(torch.mean(emd))
    emd_list.append(time() - clock)
    
    clock = time()
    cd, _ = calc_cd(points1, points2)
    cd_val.append(torch.mean(cd))
    cd_list.append(time() - clock)

print(f'DCD finshed in: {sum(dcd) / len(dcd)}. Avg. value = {sum(dcd_val) / len(dcd_val)}')
print(f'EMD finshed in: {sum(emd) / len(emd)}. Avg. value = {sum(emd_val) / len(emd_val)}')
print(f'CD finshed in: {sum(cd) / len(cd)}. Avg. value = {sum(cd_val) / len(cd_val)}')
