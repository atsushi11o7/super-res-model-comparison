import torch
from torch import nn

class ESPCN(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x