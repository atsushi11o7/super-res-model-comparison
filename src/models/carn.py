import torch
import torch.nn as nn
from src.models.modules import ResidualBlock


class CARN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=3, mid_channels=64, upscale_factor=4):
        super(CARN, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(mid_channels, mid_channels, 3) for _ in range(num_blocks)
        ])
        
        # Upsampling layers
        self.upscale = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        
        for block in self.res_blocks:
            out = block(out)
        
        out = self.upscale(out)
        return out
