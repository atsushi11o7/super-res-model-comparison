import torch
import torch.nn as nn
import torch.nn.functional as F


class VDSR(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, depth=20):
        super(VDSR, self).__init__()

        layers = []
        layers.append(nn.Conv2d(num_channels, 64, kernel_size=3, padding=3//2))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=3//2))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=3//2))
        self.net = nn.Sequential(*layers)

        self.scale_factor = scale_factor

    def forward(self, x):
        x_upscaled = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        # 残差接続
        residual = self.net(x_upscaled)
        outputs = x_upscaled + residual
        return outputs