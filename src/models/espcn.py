import torch
from torch import nn

from src.models.modules import ResidualBlock

class ESPCN(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
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

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

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


class EnhancedESPCNWithResiduals(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        # ResidualBlock
        # self.res_block = ResidualBlock(in_channels=64, kernel_size=3)

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        # ResidualBlock
        self.res_block1 = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)
        #self.res_block2 = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)

        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        
        # ResidualBlock
        #x = self.res_block(x)
        
        x = self.act(self.conv_2(x))

        # ResidualBlock
        x = self.res_block1(x)
        #x = self.res_block2(x)

        x = self.conv_3(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x


class DenseESPCN(nn.Module):
    def __init__(self, scale_factor=4, num_dense_layers=2, growth_rate=8):
        super(DenseESPCN, self).__init__()
        self.scale_factor = scale_factor

        # 初期の畳み込み層を1層に減らす
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        # Dense Blockの追加
        self.dense_block = DenseBlock(in_channels=32, num_layers=num_dense_layers, growth_rate=growth_rate)

        # 最終的な畳み込み層を残す
        self.conv_2 = nn.Conv2d(in_channels=self.dense_block.out_channels, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.dense_block(x)  # Dense Blockを通過
        x = self.conv_2(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x