import torch
import torch.nn as nn

class IMDBlock(nn.Module):
    def __init__(self, in_channels):
        super(IMDBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=True)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        distilled_out = self.conv3(out)
        remaining_out = out - distilled_out
        
        return distilled_out, remaining_out

class IMDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mid_channels=64, num_blocks=6, upscale_factor=4):
        super(IMDN, self).__init__()
        
        # Initial convolution layer
        self.entry_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        
        # Information Multi-Distillation Blocks
        self.blocks = nn.ModuleList([IMDBlock(mid_channels) for _ in range(num_blocks)])
        
        # Fusion and reconstruction layers
        self.fusion_conv = nn.Conv2d(mid_channels * num_blocks, mid_channels, kernel_size=1, padding=0, bias=True)
        self.reconstruction_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.entry_conv(x)
        
        distilled_outputs = []
        for block in self.blocks:
            distilled_out, x = block(x)
            distilled_outputs.append(distilled_out)
        
        fused_out = torch.cat(distilled_outputs, dim=1)
        fused_out = self.fusion_conv(fused_out)
        
        out = fused_out + self.reconstruction_conv(fused_out)
        out = self.upsample(out)
        
        return out