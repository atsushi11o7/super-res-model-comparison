import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=1):
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
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1,)