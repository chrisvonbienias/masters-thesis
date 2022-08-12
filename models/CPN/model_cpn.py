import sys

import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points
from torchsummary import summary

from pvconv import PVConv
from shared_mlp import SharedMLP

sys.path.append("/home/kbieniek/Desktop/masters-thesis/utils/expansion_penalty")
import expansion_penalty_module as expansion
sys.path.append("/home/kbieniek/Desktop/masters-thesis/utils/MDS")
import MDS_module


class ResidualNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.pvconv = nn.ModuleList((
            PVConv(6, 16, kernel_size=3, resolution=4),
            PVConv(16, 32, kernel_size=3, resolution=4),
            PVConv(32, 64, kernel_size=3, resolution=8),
            PVConv(64, 128, kernel_size=3, resolution=8),
            PVConv(128, 256, kernel_size=3, resolution=16),
            PVConv(256, 512, kernel_size=3, resolution=16)
        ))
        
        self.mpl = nn.Sequential(
            SharedMLP(1008, 512),
            SharedMLP(512, 256),
            SharedMLP(256, 6),
        )
        
    
    def forward(self, input):
        output = []
        input = input.transpose(2, 1)
        features = input
        coords = input[:, :3, :]
        for i in range(0, len(self.pvconv)):
            features, coords = self.pvconv[i]((features, coords))
            output.append(features)
            
        output = torch.cat(output, dim=1)
        output = self.mpl(output)
        output = torch.cat((input, output), dim=2)
        
        return output.transpose(2, 1)


class MorphingDecoder(nn.Module):
    def __init__(self, latent_dim=1024) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, 1, bias=False),
            nn.ELU(),
            nn.Conv1d(self.latent_dim, self.latent_dim // 2, 1, bias=False),
            nn.ELU(),
            nn.Conv1d(self.latent_dim // 2, self.latent_dim // 4, 1, bias=False),
            nn.ELU(),
            nn.Conv1d(self.latent_dim // 4, 6, 1, bias=False),
            nn.ELU(),
        )
        
        
    def forward(self, input):
        return self.decoder(input)


class CPN(nn.Module):
    def __init__(self, num_dense=4096, latent_dim=1024, num_primitives=8) -> None:
        super().__init__()
        
        self.num_dense = num_dense
        self.num_coarse = num_dense // 4
        self.latent_dim = latent_dim
        self.num_primitives = num_primitives
        
        self.encoder = nn.Sequential(
            PVConv(6, 64, kernel_size=3, resolution=16),
            PVConv(64, 128, kernel_size=3, resolution=16),
            PVConv(128, 512, kernel_size=3, resolution=8),
            PVConv(512, self.latent_dim, kernel_size=3, resolution=4)
        )
        
        self.linear = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ELU()
        )
        
        self.decoder = nn.ModuleList(MorphingDecoder(self.latent_dim + 2) for i in range(0, self.num_primitives))
        self.expansion = expansion.expansionPenaltyModule()
        self.residual_network = ResidualNetwork()
        
        
    def forward(self, xyz):
        B, N, C = xyz.shape
        
        # ENCODER
        xyz = xyz.transpose(2, 1)
        features, _ = self.encoder((xyz, xyz[:, :3, :]))
        features = torch.max(features, dim=2)[0]
        features = self.linear(features)
        
        # DECODER
        outs = []
        for i in range(0, self.num_primitives):
            rand_grid = torch.randn((B, 2, self.num_coarse // self.num_primitives), dtype=torch.float32, device='cuda:0')
            y = features.unsqueeze(2).expand(features.size(0),features.size(1), rand_grid.size(2))
            y = torch.cat((rand_grid, y), dim=1)
            outs.append(self.decoder[i](y))
            
        outs = torch.cat(outs, 2)
        coarse = outs.transpose(2, 1) # (B, COARSE, 6)
        
        # EXPANSION LOSS
        dist, _, mean_mst_dis = self.expansion(coarse, self.num_coarse // self.num_primitives, 1.5)
        exp_loss = torch.mean(dist)
        
        # MINIMUM DENSITY SAMPLE
        dense = torch.cat((xyz.transpose(2, 1), coarse), dim=1)
        resampled_idx = MDS_module.minimum_density_sample(dense.contiguous(), dense.shape[1], mean_mst_dis) 
        dense = MDS_module.gather_operation(dense, resampled_idx)
        
        # RESIDUAL NETWORK
        dense = self.residual_network(dense)
        dense = sample_farthest_points(coarse, K=self.num_dense)[0] # (B, DENSE, 6)
        
        return coarse, dense, exp_loss
    
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CPN().to(device)
    print(model)