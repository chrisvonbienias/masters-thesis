import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, inputs):
        return self.layers(inputs)
