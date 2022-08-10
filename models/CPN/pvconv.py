import torch.nn as nn

import functional as F
from voxelization import Voxelization
from shared_mlp import SharedMLP


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution)
        self.voxel_layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
        )
        self.point_features = SharedMLP(in_channels, out_channels)


    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        
        return fused_features, coords
