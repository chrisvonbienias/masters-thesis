import torch
import torch.nn as nn
from torchsummary import summary


class PointAutoEncoder(nn.Module):
    def __init__(self, num_dense=2048):
        super().__init__()
        
        self.num_dense = num_dense
        
        self.conv2d_64a = nn.Sequential(
            nn.Conv2d(self.num_dense, 64, kernel_size=(6, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2d_64b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2d_128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv2d_1024 = nn.Sequential(
            nn.Conv2d(128, 1024, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_dense * 6),
            nn.BatchNorm1d(self.num_dense * 6),
            nn.ReLU(inplace=True),
        )
        
        
    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        input = torch.unsqueeze(xyz, -1)
        
        # encoder
        net = self.conv2d_64a(input)
        net = self.conv2d_64b(net)
        point_feat = self.conv2d_64b(net)
        net = self.conv2d_128(point_feat)
        global_feat = self.conv2d_1024(net)
        global_feat = torch.max(global_feat, dim=-1)[0]
        global_feat = torch.max(global_feat, dim=-1)[0]
        
        # decoder
        net = self.decoder(global_feat)
        net = torch.reshape(net, (B, N, 6))
        
        return global_feat.contiguous(), net.contiguous()
        
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointAutoEncoder().to(device)
    summary(model, (2048, 6))
