import torch
import torch.nn as nn

from pvconv import PVConv
from shared_mlp import SharedMLP


class CPN(nn.Module):
    def __init__(self, num_dense=8192, latent_dim=1024) -> None:
        super().__init__()
        
        self.num_dense = num_dense
        self.num_coarse = num_dense // 4
        self.latent_dim = latent_dim
        
        self.pv_conv = nn.Sequential(
            PVConv(6, 2048, kernel_size=3, resolution=16),
            PVConv(2048, 1024, kernel_size=3, resolution=16),
            PVConv(1024, 512, kernel_size=3, resolution=8)
        )
        
        self.shared_mlp = nn.Sequential(
            SharedMLP(6, 2048),
            SharedMLP(2048, 1024),
            SharedMLP(1024, 512),
        )
        
        
    def forward(self, xyz):
        B, N, C = xyz.shape
        
        # ENCODER
        xyz = xyz.transpose(2, 1)
        features1, _ = self.pv_conv((xyz, xyz[:, :3, :]))
        features2 = self.shared_mlp(xyz)
        features = torch.cat((features1, features2), dim=1)
        features = torch.max(features, dim=2)[0]
        
        # DECODER
        
        return features
    
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt = torch.randn((32, 2048, 6), device=device)
    model = CPN(4096).to(device)
    res = model(pt)