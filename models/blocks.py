import torch.nn as nn


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Spectral3DCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, kernel_size=3, stride=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv3DBlock(in_channels if i == 0 else out_channels,
                        out_channels, kernel_size, stride)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
