import sys

import torch
import torch.nn as nn
from torchsummary import summary

sys.path.append("/home/kbieniek/Desktop/masters-thesis/models/CPN")
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
    def __init__(self, num_dense=4096, latent_dim=1024, num_primitives=8, num_cat=47) -> None:
        super().__init__()
        
        self.num_dense = num_dense
        self.num_coarse = num_dense // 4
        self.latent_dim = latent_dim
        self.num_primitives = num_primitives
        self.num_cat = num_cat + 1
        
        self.encoder = nn.Sequential(
            PVConv(6, 256, kernel_size=3, resolution=16),
            PVConv(256, 512, kernel_size=3, resolution=8),
            PVConv(512, self.latent_dim // 2, kernel_size=3, resolution=4),
        )
        
        self.label_encoder = nn.Sequential(
            nn.Embedding(self.num_cat, 3, padding_idx=0),
            nn.Flatten(),
            nn.Linear(self.num_cat * 3, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 512)
        )
        
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.latent_dim // 2, self.latent_dim // 2),
            nn.ELU()
        )
        
        self.decoder = nn.ModuleList(MorphingDecoder(self.latent_dim + 2) for i in range(0, self.num_primitives))
        self.expansion = expansion.expansionPenaltyModule()
        self.residual_network = ResidualNetwork()
        
        
    def forward(self, xyz):
        label, pt = xyz
        B, N, C = pt.shape
        
        # ENCODER
        pt = pt.transpose(2, 1)
        pt_features, _ = self.encoder((pt, pt[:, :3, :]))
        pt_features = torch.max(pt_features, dim=2)[0]
        pt_features = self.linear(pt_features)
        
        label_feat = self.label_encoder(label)
        features = torch.cat((pt_features, label_feat), dim=1)
        
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
        dense = torch.cat((pt.transpose(2, 1), coarse), dim=1)
        resampled_idx = MDS_module.minimum_density_sample(dense.contiguous(), 2048, mean_mst_dis) 
        dense = MDS_module.gather_operation(dense.transpose(2, 1).contiguous(), resampled_idx)
        
        # RESIDUAL NETWORK
        dense = self.residual_network(dense.transpose(2, 1)) # (B, DENSE, 6)
        
        return coarse, dense, exp_loss
    

class CPN_Discriminator(nn.Module):
    def __init__(self, num_dense=4096, latent_dim=1024) -> None:
        super().__init__()
        
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        
        self.network = nn.Sequential(
            PVConv(6, 128, 16),
            PVConv(128, 256, 16),
            PVConv(256, 512, 8),
            PVConv(512, self.latent_dim, 8),
            nn.Dropout(0.2),
            nn.Linear(self.latent_dim, 512),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1)
        )
        
        
    def forward(self, input):
        return self.network(input)
            
            
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CPN().to(device)
    pt = torch.randn((8, 2048, 6), device=device)
    label = torch.randint(0, 2, (8, 48), device=device)
    model((label, pt))