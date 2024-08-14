import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3):
        super(SRCNN, self).__init__()

        self.scale_factor = scale_factor

        # Patch Extraction and Representation Layer
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, padding=9//2, padding_mode='replicate')

        # Non-linear Mapping Layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)

        # Reconstruction Layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=num_channels, kernel_size=5, padding=5//2, padding_mode='replicate')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class FSRCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        
        self.feature_extraction = nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2, bias=True)
        
        self.shrinking = nn.Conv2d(d, s, kernel_size=1, padding=0, bias=True)
        
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1, bias=True))
            mapping_layers.append(nn.ReLU(inplace=True))
        self.mapping = nn.Sequential(*mapping_layers)
        
        self.expanding = nn.Conv2d(s, d, kernel_size=1, padding=0, bias=True)
        
        self.deconvolution = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.feature_extraction(x))
        x = self.relu(self.shrinking(x))
        x = self.mapping(x)
        x = self.relu(self.expanding(x))
        x = self.deconvolution(x)
        return x
