import torch
from torch import nn

from src.models.modules import ResidualBlock, SimpleResBlock, ChannelAttention, DenseBlock

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


class ESPCNWithResidualBlock(nn.Module):
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

        self.res_block = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)

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
        x = self.res_block(x)
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x
    

class ESPCNWithResidualBlockV2(nn.Module):
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

        # Residual block instead of simple conv layer
        self.res_block = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.res_block(x)  # Pass through the residual block
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x
    

class ESPCNWithResidualBlockV3(nn.Module):
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

        # Residual block instead of simple conv layer

        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.res_block = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.res_block(x)  # Pass through the residual block
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x

    
class ESPCNWithSimpleResBlock(nn.Module):
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

        # Use SimpleResBlock instead of a single convolutional layer
        self.simple_res_block = SimpleResBlock(in_channels=32, kernel_size=3)
        
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
        x = self.simple_res_block(x)  # Pass through the SimpleResBlock
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x
    

class ESPCNWithResBlockV2AndAttention2(nn.Module):
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

        # First Channel Attention after conv_2
        self.channel_attention_1 = ChannelAttention(in_channels=32, reduction_ratio=8)

        # Residual block instead of simple conv layer
        self.res_block = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)

        # Second Channel Attention after ResidualBlock
        self.channel_attention_2 = ChannelAttention(in_channels=32, reduction_ratio=8)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.channel_attention_1(x)  # Apply first channel attention
        x = self.res_block(x)  # Pass through the residual block
        x = self.channel_attention_2(x)  # Apply second channel attention
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x
    

class ESPCNWithResBlockAndAttention(nn.Module):
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

        self.res_block = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)

        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.channel_attention = ChannelAttention(in_channels=32, reduction_ratio=8)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.res_block(x)
        x = self.act(self.conv_3(x))
        x = self.channel_attention(x)
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x



class ESPCNWithDense(nn.Module):
    def __init__(self, scale_factor=4, num_dense_layers=2, growth_rate=8):
        super(ESPCNWithDense, self).__init__()
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


class ESPCNWithDenseAndChannelAttention(nn.Module):
    def __init__(self, scale_factor=4, num_dense_layers=2, growth_rate=8, reduction=16):
        super(ESPCNWithDenseAndChannelAttention, self).__init__()
        self.scale_factor = scale_factor

        # 初期の畳み込み層 (RGB画像に対応するため、in_channels=3)
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        # Dense Blockを初期畳み込み層の後に追加
        self.dense_block = DenseBlock(in_channels=32, num_layers=num_dense_layers, growth_rate=growth_rate)

        # チャネルアテンションをDense Blockの後に追加
        self.channel_attention = ChannelAttention(in_channels=self.dense_block.out_channels, reduction=reduction)

        # 中間の畳み込み層
        self.conv_2 = nn.Conv2d(in_channels=self.dense_block.out_channels, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        # 最終的な畳み込み層 (RGB画像のため、out_channels=3*scale_factor^2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=(3 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_1(x))
        x = self.dense_block(x)  # Dense Blockを通過
        x = self.channel_attention(x)  # チャネルアテンションを適用
        x = self.act(self.conv_2(x))
        x = self.conv_3(x)
        x = self.pixel_shuffle(x)
        x = torch.clip(x, 0.0, 1.0)
        return x
    

class ESPCNWithPixelShuffle2(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.Tanh()

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        half_scale_factor = scale_factor // 2

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(32 * half_scale_factor * half_scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.conv_5 = nn.Conv2d(in_channels=32, out_channels=(num_channels * half_scale_factor * half_scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_5.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_5.bias)

        self.pixel_shuffle = nn.PixelShuffle(half_scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        x = self.conv_5(x)
        x = self.pixel_shuffle(x)
        x = torch.clip(x, 0.0, 1.0)
        return x
    

class ESPCN2x2(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        half_scale_factor = 2

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(num_channels * half_scale_factor * half_scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.conv_5 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_5.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_5.bias)

        self.conv_6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_6.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_6.bias)

        self.conv_7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_7.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_7.bias)

        self.conv_8 = nn.Conv2d(in_channels=32, out_channels=(num_channels * half_scale_factor * half_scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_8.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_8.bias)

        self.pixel_shuffle = nn.PixelShuffle(half_scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = torch.clip(x, 0.0, 1.0)
        tmp_output = x.clone()
        x = self.act(self.conv_5(x))
        x = self.act(self.conv_6(x))
        x = self.act(self.conv_7(x))
        x = self.conv_8(x)
        x = self.pixel_shuffle(x)
        x = torch.clip(x, 0.0, 1.0)
        return x, tmp_output
    

class EnhancedESPCN(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3_1.bias)
        
        self.res_block = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)  # 残差ブロックを追加

        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.act(self.conv_3_1(x))  # 追加した層
        x = self.res_block(x)  # 残差ブロックの適用
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x
    

class ESPCNWithResidualBlockTTA(nn.Module):
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

        self.res_block = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)

        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale_factor * self.scale_factor), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input :  (1, C, H, W)
        x_augmented = self.apply_augmentations(x)  # (8, C, H, W)

        x_augmented = self.super_resolve(x_augmented)  # (8, C, H*scale, W*scale)

        x_restored = self.reverse_and_average(x_augmented)  # (1, C, H*scale, W*scale)

        return x_restored

    def super_resolve(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.res_block(x)
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clip(x, 0.0, 1.0)
        return x

    def apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 8 types of augmentations to the input image for Test-Time Augmentation (TTA).
        - Original image
        - 90-degree, 180-degree, and 270-degree rotations
        - Horizontally flipped original image
        - 90-degree, 180-degree, and 270-degree rotations of the horizontally flipped image

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: A batch of augmented images concatenated along the batch dimension.
        """

        augmentations = [x]
        
        augmentations.extend([self.rotate_90(x, k) for k in range(1, 4)])
        #augmentations.extend([self.rotate_90(x, 2)])
        
        flipped_x = torch.flip(x, [3])
        augmentations.append(flipped_x)
        
        augmentations.extend([self.rotate_90(flipped_x, k) for k in range(1, 4)])
        #augmentations.extend([self.rotate_90(flipped_x, 2)])

        return torch.cat(augmentations, dim=0)

    def reverse_and_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        各拡張の逆変換を行い、平均化します。
        """
        #"""
        x_reversed = [
            x[0],  # 元の画像
            self.rotate_90(x[1], 3),  # 90度回転の逆変換（逆方向に270度回転）
            self.rotate_90(x[2], 2),  # 180度回転の逆変換（逆方向に180度回転）
            self.rotate_90(x[3], 1),  # 270度回転の逆変換（逆方向に90度回転）
            torch.flip(x[4], [2]),  # 水平反転の逆変換（再度水平反転）
            torch.flip(self.rotate_90(x[5], 3), [2]),  # 90度回転+水平反転の逆変換
            torch.flip(self.rotate_90(x[6], 2), [2]),  # 180度回転+水平反転の逆変換
            torch.flip(self.rotate_90(x[7], 1), [2])   # 270度回転+水平反転の逆変換
        ]
        """
        x_reversed = [
            x[0],  # 元の画像
            self.rotate_90(x[1], 2),  # 180度回転の逆変換（逆方向に180度回転）
            torch.flip(x[2], [2]),  # 水平反転の逆変換（再度水平反転）
            torch.flip(self.rotate_90(x[3], 2), [2]),  # 180度回転+水平反転の逆変換
        ]
        """

        x_avg = torch.mean(torch.stack(x_reversed, dim=0), dim=0, keepdim=True)
        return x_avg

    
    def rotate_90(self, tensor, k):
        """
        A function to manually implement a 90-degree rotation.

        Args:
            tensor (torch.Tensor): The tensor to rotate.
            k (int): The number of times to rotate by 90 degrees.

        Returns:
            torch.Tensor: The tensor after rotation.
        """
        if tensor.dim() == 4:
            # 4D tensor: (1, C, W, H)
            if k == 1:
                return tensor.transpose(2, 3).flip(2)
            elif k == 2:
                return tensor.flip(2).flip(3)
            elif k == 3:
                return tensor.transpose(2, 3).flip(3)
            else:
                return tensor
        elif tensor.dim() == 3:
            # 3D tensor: (C, W, H)
            if k == 1:
                return tensor.transpose(1, 2).flip(1)
            elif k == 2:
                return tensor.flip(1).flip(2)
            elif k == 3:
                return tensor.transpose(1, 2).flip(2)
            else:
                return tensor
        else:
            raise ValueError("Unsupported tensor dimensions. Tensor must be 3D or 4D.")