from matplotlib.pyplot import show
from tqdm import tqdm
import plyfile
import numpy as np
import open3d as o3d
from skimage.color import rgb2lab, lab2rgb

import torch
from torch.utils.data.dataloader import DataLoader

from models import PCN, FinerPCN, MSN, CRN_Generator
from dataset import ShapeNet
from train import NUM_DENSE
from utils.model_utils import calc_dcd, calc_cd, calc_emd

BATCH_SIZE = 32
MODEL_NAME = 'CRN'
MODEL_PATH = 'checkpoint/CRN/CRN_GEN_DCD_8_2_14:4.pth'
NUM_DENSE = 8192

def read_point_cloud(path):
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
    
    
def random_sample(pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
    
    
def test() -> None:
    
    data_path = "./resources/dataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # dataset and dataloader
    test_seen_dataset = ShapeNet(data_path, split='test_seen', num_dense=NUM_DENSE)
    test_seen_dataloader = DataLoader(test_seen_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_unseen_dataset = ShapeNet(data_path, split='test_unseen', num_dense=NUM_DENSE)
    test_unseen_dataloader = DataLoader(test_unseen_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print('Dataset ready!')
    
    # model
    if MODEL_NAME == 'PCN':
        model = PCN(num_dense=NUM_DENSE, latent_dim=1024, grid_size=4).to(device)
    elif MODEL_NAME == 'FinerPCN':
        model = FinerPCN(num_dense=NUM_DENSE, grid_size=4, grid_scale=0.5).to(device)
    elif MODEL_NAME == 'MSN':
        model = MSN(num_points=NUM_DENSE).to(device)
    elif MODEL_NAME == 'CRN':
        model = CRN_Generator(num_dense=NUM_DENSE).to(device)
    else:
        raise ValueError(f'Model {MODEL_NAME} not implemented')
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print('Model loaded!')
    
    seen_loss_cd = 0
    seen_loss_dcd = 0
    seen_loss_emd = 0
    unseen_loss_cd = 0
    unseen_loss_dcd = 0
    unseen_loss_emd = 0
    with torch.no_grad():
        
        i = 0
        for c, p in tqdm(test_seen_dataloader, desc='Testing seen...'):
            c = c.to(device)
            p = p.to(device)
            
            _, dense_pred, *other = model(p)
            
            seen_loss_cd += torch.mean(calc_cd(dense_pred, c)[0]).item()
            seen_loss_dcd += torch.mean(calc_dcd(dense_pred, c)[0]).item()
            seen_loss_emd += (torch.mean(calc_emd(dense_pred[:, :, :3], c[:, :, :3])).item() + torch.mean(calc_emd(dense_pred[:, :, 3:], c[:, :, 3:])).item()) / 2
            i += 1
            
        seen_loss_cd = seen_loss_cd / i
        seen_loss_dcd = seen_loss_dcd / i
        seen_loss_emd = seen_loss_emd / i
        i = 0
        
        for c, p in tqdm(test_unseen_dataloader, desc='Testing unseen...'):
            c = c.to(device)
            p = p.to(device)
            
            _, dense_pred, *other = model(p)
            
            unseen_loss_cd += torch.mean(calc_cd(dense_pred, c)[0]).item()
            unseen_loss_dcd += torch.mean(calc_dcd(dense_pred, c)[0]).item()
            unseen_loss_emd += (torch.mean(calc_emd(dense_pred[:, :, :3], c[:, :, :3])).item() + torch.mean(calc_emd(dense_pred[:, :, 3:], c[:, :, 3:])).item()) / 2
            i += 1
            
        unseen_loss_cd = unseen_loss_cd / i
        unseen_loss_dcd = unseen_loss_dcd / i
        unseen_loss_emd = unseen_loss_emd / i
            
    print(f'Seen loss | CD: {seen_loss_cd} | DCD: {seen_loss_dcd} | EMD: {seen_loss_emd} |')
    print(f'Unseen loss | CD: {unseen_loss_cd} | DCD: {unseen_loss_dcd} | EMD: {unseen_loss_emd} |')

    
def show_examples():

    datapath = 'resources/dataset/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    seen_examples_list = [
        "1021a0914a7207aff927ed529ad90a11.ply",  # airplane
        "1a1b62a38b2584874c62bee40dcdc539.ply",  # cabinet
        "1a0bc9ab92c915167ae33d942430658c.ply",  # car
        "1a6f615e8b1b5ae4dbbc9440457e303e.ply",  # chair
        "1a5ebc8575a4e5edcc901650bbbbb0b5.ply",  # lamp
        "1a4a8592046253ab5ff61a3a2a0e2484.ply",  # sofa
        "1a00aa6b75362cc5b324368d54a7416f.ply",  # table
        "1a2b1863733c2ca65e26ee427f1e5a4c.ply",  # boat
        ]  
    
    unseen_examples_list = [
        "1aae0b0836cd03ab90b756c60eaa9646.ply",  # bus
        "1aa55867200ea789465e08d496c0420f.ply",  # bed
        "1ab8202a944a6ff1de650492e45fb14f.ply",  # bookshelf
        "1a40eaf5919b1b3f3eaa2b95b99dae6.ply",   # bench
        "1a96f73d0929bd4793f0194265a9746c.ply",  # guitar
        "1a2d2208f73d0531cec33e62192b66e5.ply",  # motorbike
        "1ad8d7573dc57ebc247a956ed3779e34.ply",  # skateboard
        "1a640c8dffc5d01b8fd30d65663cfd42.ply",  # pistol
        ] 

    seen_clouds = []
    for cloud in seen_examples_list:
        pc = read_point_cloud(f'{datapath}/partial/{cloud}')
        seen_clouds.append(pc)
        
    unseen_clouds = []
    for cloud in unseen_examples_list:
        pc = read_point_cloud(f'{datapath}/partial/{cloud}')
        unseen_clouds.append(pc)
        
    # model
    if MODEL_NAME == 'PCN':
        model = PCN(num_dense=NUM_DENSE, latent_dim=1024, grid_size=4).to(device)
    elif MODEL_NAME == 'FinerPCN':
        model = FinerPCN(num_dense=NUM_DENSE, grid_size=4, grid_scale=0.5).to(device)
    else:
        raise ValueError(f'Model {MODEL_NAME} not implemented')
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    for pt in seen_clouds:
        pt = random_sample(pt, 2048)
        pt = torch.from_numpy(pt)
        pt = torch.unsqueeze(pt, dim=0)
        pt = pt.to(device)
        _, dense = model(pt)
        dense = dense.detach().cpu().numpy()[0, :, :]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(dense[:,3:6])
        o3d.visualization.draw_geometries([pcd])
        
        
if __name__ == '__main__':
    test()
    # show_examples()