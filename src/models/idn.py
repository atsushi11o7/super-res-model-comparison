import torch
import torch.nn as nn


class DenselyConnectedBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenselyConnectedBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = torch.cat([x, out], 1)
        return out
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

class IDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=4, growth_rate=16, upscale_factor=4):
        super(IDN, self).__init__()
        
        # Initial feature extraction layer
        self.entry_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Densely connected blocks
        self.blocks = nn.ModuleList([DenselyConnectedBlock(num_features + i * growth_rate, growth_rate) for i in range(num_blocks)])
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(num_features + num_blocks * growth_rate, num_features, kernel_size=1, padding=0, bias=True)
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        x = self.entry_conv(x)
        x = self.relu(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.fusion_conv(x)
        x = self.upsample(x)
        
        return x
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.entry_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.entry_conv.bias, 0)
        nn.init.kaiming_normal_(self.fusion_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fusion_conv.bias, 0)
        for m in self.upsample:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)