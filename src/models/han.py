import torch
import torch.nn as nn
import torch.nn.init as init
from src.models.modules import ResidualBlock


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.sigmoid(out)
        return x * out
    
    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)

class HAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mid_channels=64, num_blocks=8, upscale_factor=4):
        super(HAN, self).__init__()
        
        # Initial convolution layer
        self.entry_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels, 3) for _ in range(num_blocks)])
        
        # Attention block
        self.attention = AttentionBlock(mid_channels)
        
        # Fusion and reconstruction layers
        self.fusion_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.reconstruction_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        x = self.entry_conv(x)
        x = self.relu(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.attention(x)
        
        fused_out = self.fusion_conv(x)
        out = fused_out + self.reconstruction_conv(fused_out)
        out = self.upsample(out)
        
        return out
    
    def _initialize_weights(self):
        init.kaiming_normal_(self.entry_conv.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.entry_conv.bias, 0)
        init.kaiming_normal_(self.fusion_conv.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fusion_conv.bias, 0)
        init.kaiming_normal_(self.reconstruction_conv.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.reconstruction_conv.bias, 0)
        for m in self.upsample:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
