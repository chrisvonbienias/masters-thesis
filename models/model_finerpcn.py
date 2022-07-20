from math import dist
from threading import local
import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather

class FinerPCN(nn.Module):
    
    def __init__(self, num_dense=8192, grid_size=4, grid_scale=0.5):
        super().__init__()
        
        self.num_dense = num_dense
        self.grid_size = grid_size
        self.grid_scale = grid_scale
        self.num_coarse = self.num_dense // (self.grid_size ** 2)
        
        self.mlpConv256 = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        self.mlpConv512 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.mlpConv1024 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )
        
        self.mlpConvFinal = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 48, 1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 6 * self.num_coarse)
        )
        
        
    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder 1
        feature = self.mlpConv256(xyz.transpose(2, 1))                                       # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B, 512, N)
        feature = self.mlpConv1024(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder 1
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 6)                    # (B, 512, N)
        coarse = torch.cat([coarse, xyz], dim=1)                                             # (B, 512+2048, N)
        
        # farthest point sampling
        coarse2 = sample_farthest_points(coarse, K=1024)[0]                                  # (B, 1024, 6)
        
        # local density
        distance = knn_points(coarse2, coarse2, K=16)                                          
        density = knn_gather(coarse2, distance[1])                                           # (B, 1024, 16, 6)
        density = torch.exp(-density / (0.34 ** 2))
        density = torch.mean(density, dim=2)                                                 # (B, 1024, 6)
        density = self.mlpConv256(density.transpose(2, 1))                                   # (B, 256, 1024)
        
        # local patches
        local_patches = knn_gather(coarse2, distance[1])                                     # (B, 1024, 16, 6)
        local_patches = local_patches - torch.unsqueeze(coarse2, 2)                          # (B, 1024, 16, 6)
        local_patches = torch.mean(local_patches, 2)                                         # (B, 1024, 6)
        local_patches = self.mlpConv256(local_patches.transpose(2, 1))                       # (B, 256, 1024)
        
        # folding 1
        folding = self.mlpConv256(coarse2.transpose(2, 1))                                   # (B, 256, 1024)
        folding = torch.multiply(folding, density)                                           # (B, 256, 1024)
        folding = torch.multiply(folding, local_patches)                                     # (B, 256, 1024)
        folding = torch.sum(folding, dim=2)                                                  # (B, 256)
        
        # encoder 2
        feature = self.mlpConv256(coarse2.transpose(2, 1))                                   # (B, 256, 1024)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, 1024), feature], dim=2)           # (B, 512, 2048)
        
        # encoder 3
        feature = self.mlpConv512(feature)                                                   # (B, 512, 2048)
        feature = torch.max(feature, dim=2, keepdim=True)[0]                                 # (B, 512, 1)
        
        # folding 2
        folding = folding.unsqueeze(dim=2)                                                   # (B, 256, 1)
        folding = folding.expand(-1, -1, 256)                                                # (B, 256, 256)
        folding = self.mlpConv512(folding)                                                   # (B, 512, 256)
        folding = torch.max(folding, dim=2, keepdim=True)[0]                                 # (B, 512, 1)
        folding = torch.cat([folding, feature], dim=1)                                       # (B, 512 + 512, 1)
        
        # folding 3
        folding = self.mlpConvFinal(folding.expand(-1, -1, 1024))                            # (B, 24, 1024)
        fine = torch.reshape(folding, (B, self.num_dense, 6))
        
        return coarse.contiguous(), fine.contiguous()
        