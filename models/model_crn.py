from math import log2, sqrt
from typing import List
from debugpy import debug_this_thread

import torch
import torch.nn as nn

from torchsummary import summary
from pytorch3d.ops import sample_farthest_points, ball_query, knn_gather


class CRN_Generator(nn.Module):
    def __init__(self, num_dense=8192, num_coarse=1024, latent_dim=1024):
        super().__init__()
        
        self.num_dense = num_dense
        self.num_coarse = num_coarse
        self.latent_dim = latent_dim
        self.feature_size = torch.load('checkpoint/PointAutoEncoder/mean_features.pt')
        self.ratio = num_dense / num_coarse / 2
        
        self.mlp_conv128_256 = nn.Sequential(
            nn.Conv1d(6, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        self.mlp_conv512_lat = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )
            
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_coarse * 6),
        )
        
        self.mlp_conv128_64 = nn.Sequential(
            nn.Conv1d(1160, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(64, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 6, 1)
        )
        
        self.features = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        
        self.contract1 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), padding='valid'),
            nn.ReLU(inplace=True)
        )
        
        self.contract2 = nn.Sequential(
            nn.Conv2d(128, 64, (1, 1), padding='valid'),
            nn.ReLU(inplace=True)
        )
        
        self.expand = nn.Sequential(
            nn.Conv2d(64, 128, (1, 1), padding='valid'),
            nn.ReLU(inplace=True)
        )
        
    
    def generate_grid(self, ratio):
        sqrted = int(sqrt(ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (ratio % i) == 0:
                num_x = i
                num_y = ratio // 1
                break
        grid_x = torch.linspace(-0.2, 0.2, num_x)
        grid_y = torch.linspace(-0.2, 0.2, num_y)
        x, y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid = torch.reshape(torch.stack([x, y], dim=-1), (-1, 2))
        return grid.cuda()
    
    
    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        features = self.mlp_conv128_256(xyz.transpose(2, 1))                         # (B, 256, N)
        features_global = torch.max(features, dim=2, keepdim=True)[0]                # (B, 256, 1)
        features = torch.cat([features, features_global.expand(-1, -1, N)], dim=1)   # (B, 512, N)
        features = self.mlp_conv512_lat(features)                                    # (B, latent_dim, N)
        features = torch.max(features, dim=2)[0]                                     # (B, latent_dim)
        
        # decoder
        
        # coarse
        coarse = self.mlp(features)                                                  # (B, num_coarse * 6)
        coarse = torch.tanh(coarse)
        coarse = torch.reshape(coarse, (-1, self.num_coarse, 6))                     # (B, num_coarse, 6)
        
        # mirror + fps
        input_fps = sample_farthest_points(xyz, K=self.num_coarse // 2)[0]           # (B, num_coarse / 2, 6)
        input_fps_filp = torch.cat(
            [torch.unsqueeze(input_fps[:, :, 0], dim=2),
             torch.unsqueeze(input_fps[:, :, 1], dim=2),
             torch.unsqueeze(-input_fps[:, :, 2], dim=2),
             torch.unsqueeze(input_fps[:, :, 3], dim=2),
             torch.unsqueeze(input_fps[:, :, 4], dim=2),
             torch.unsqueeze(input_fps[:, :, 5], dim=2),], dim=2
        )                                                                            # (B, num_coarse / 2, 6)
        input_fps = torch.cat([input_fps, input_fps_filp], dim=1)                    # (B, num_coarse, 6)
        coarse2 = torch.cat([input_fps, coarse], dim=1)                              # (B, 2 * num_coarse, 6)                     
        
        # lifting module
        for i in range(int(log2(self.ratio))):
            num_fine = 2 ** (i + 2) * self.num_coarse
            grid = self.generate_grid(2 ** (i + 1))
            grid = torch.unsqueeze(grid, dim=0)
            grid_feat = torch.tile(grid, (B, self.num_coarse, 1))
            point_feat = torch.tile(torch.unsqueeze(coarse2, 2), (1, 1, 2, 1))
            point_feat = torch.reshape(point_feat, (-1, num_fine, 6))
            global_feat = torch.tile(torch.unsqueeze(features, 1), (1, num_fine, 1))
            
            mean_feature = self.features(self.feature_size.expand((B, -1)))
            mean_feature = torch.unsqueeze(mean_feature, dim=1)
            mean_feature = torch.tile(mean_feature, [1, num_fine, 1])
            feat = torch.cat([grid_feat, point_feat, global_feat, mean_feature], dim=2)
            
            # contract-expand operation
            feat1 = self.mlp_conv128_64(feat.transpose(2, 1))
            feat2 = torch.reshape(feat1, (feat1.shape[0], 1, -1, feat1.shape[-1]))
            feat2 = torch.transpose(feat2, 2, 1)
            feat2 = self.contract1(feat2)
            feat2 = self.expand(feat2)
            feat2 = torch.reshape(feat2, (feat2.shape[0], 128, 1, -1))
            feat2 = self.contract2(feat2)
            feat2 = torch.squeeze(feat2, dim=2)
            feat = feat1 + feat2
            
            fine = self.final_conv(feat)
            coarse2 = fine
            
        return coarse.contiguous(), fine.transpose(2, 1).contiguous()
 

class CRN_Discriminator(nn.Module):
    def __init__(self, radius_list: List[int]=[0.1, 0.2, 0.4], nsample_list: List[int]=[16, 32, 128]):
        super().__init__()
        assert len(nsample_list) == len(radius_list) 
        
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        
        self.mplConv_2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
        )
        
        self.mplConv_3 = nn.Sequential(
            nn.Conv1d(128, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 96, 1, bias=False),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(96, 128 * 6, 1, bias=False),
            nn.BatchNorm1d(128 * 6),
            nn.LeakyReLU(inplace=True),
        )
        
        self.mlpConv_4 = nn.Sequential(
            nn.Conv1d(18, 1, 1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )
    
    
    def conv1d(self, input):
            
        Conv1d = nn.Sequential(
            nn.Conv1d(input.shape[1], 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
        ).cuda()
        
        output = Conv1d(input)
        
        return output.cuda()
        
        
    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        points, _ = sample_farthest_points(xyz, K= N // 8)
        
        points_list = []
        for (radius, nsample) in zip(self.radius_list, self.nsample_list):
            new_xyz, _, _ = ball_query(points, xyz, None, None, nsample, radius)
            new_xyz = self.conv1d(new_xyz.transpose(2, 1))
            new_xyz = self.mplConv_2(new_xyz)
            new_xyz = self.mplConv_3(new_xyz)
            new_xyz = torch.reshape(new_xyz, (B, N // 8, 6, -1))
            new_xyz = torch.max(new_xyz, dim=-1)[0]
            points_list.append(new_xyz)
        
        output = torch.cat(points_list, dim=-1)
        patch_values = self.mlpConv_4(output.transpose(2,1))
        patch_values = torch.mean(patch_values, dim=-1)
        
        return patch_values.contiguous()
        
        
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CRN_Discriminator().to(device)
    summary(model, (8192, 6))
    model = CRN_Generator().to(device)
    summary(model, (2048, 6))
    