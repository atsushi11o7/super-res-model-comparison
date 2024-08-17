import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.modules import ResidualBlock, ChannelAttention


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
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=16, m=4):
        super(FSRCNN, self).__init__()
        
        # Define the layers
        self.feature_extraction = nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2, bias=True)
        self.shrinking = nn.Conv2d(d, s, kernel_size=1, padding=0, bias=True)
        
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=3//2, bias=True))
            mapping_layers.append(nn.ReLU(inplace=True))
        self.mapping = nn.Sequential(*mapping_layers)
        
        self.expanding = nn.Conv2d(s, d, kernel_size=1, padding=0, bias=True)
        self.deconvolution = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        def he_initialization(layer):
            nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(2 / (layer.out_channels * layer.weight[0][0].numel())))
            nn.init.zeros_(layer.bias)

        he_initialization(self.feature_extraction)
        he_initialization(self.shrinking)
        
        for layer in self.mapping:
            if isinstance(layer, nn.Conv2d):
                he_initialization(layer)

        he_initialization(self.expanding)
        
        nn.init.normal_(self.deconvolution.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconvolution.bias)

    def forward(self, x):
        x = self.relu(self.feature_extraction(x))
        x = self.relu(self.shrinking(x))
        x = self.mapping(x)
        x = self.relu(self.expanding(x))
        x = self.deconvolution(x)
        return x


class FSRCNNWithPixelShuffle(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN_PixelShuffle, self).__init__()
        
        self.feature_extraction = nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2, bias=True)
        nn.init.normal_(self.feature_extraction.weight, mean=0, std=0.001)
        nn.init.zeros_(self.feature_extraction.bias)
        
        self.shrinking = nn.Conv2d(d, s, kernel_size=1, padding=0, bias=True)
        nn.init.normal_(self.shrinking.weight, mean=0, std=0.001)
        nn.init.zeros_(self.shrinking.bias)
        
        mapping_layers = []
        for _ in range(m):
            conv = nn.Conv2d(s, s, kernel_size=3, padding=1, bias=True)
            nn.init.normal_(conv.weight, mean=0, std=0.001)
            nn.init.zeros_(conv.bias)
            mapping_layers.append(conv)
            mapping_layers.append(nn.ReLU(inplace=True))
        self.mapping = nn.Sequential(*mapping_layers)
        
        self.expanding = nn.Conv2d(s, d, kernel_size=1, padding=0, bias=True)
        nn.init.normal_(self.expanding.weight, mean=0, std=0.001)
        nn.init.zeros_(self.expanding.bias)
        
        self.upsample_conv = nn.Conv2d(d, num_channels * (scale_factor ** 2), kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.upsample_conv.weight, mean=0, std=0.001)
        nn.init.zeros_(self.upsample_conv.bias)

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.feature_extraction(x))
        x = self.relu(self.shrinking(x))
        x = self.mapping(x)
        x = self.relu(self.expanding(x))
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        x = torch.clip(x, 0.0, 1.0)
        return x


class FSRCNNWithAttention(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=16, m=4):
        super(FSRCNNWithAttention, self).__init__()
        
        self.feature_extraction = nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2, bias=True)
        self.shrinking = nn.Conv2d(d, s, kernel_size=1, padding=0, bias=True)

        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=3//2, bias=True))
            mapping_layers.append(nn.ReLU(inplace=True))
        self.mapping = nn.Sequential(*mapping_layers)
        
        # Channel Attention Layer
        self.channel_attention = ChannelAttention(s)

        self.expanding = nn.Conv2d(s, d, kernel_size=1, padding=0, bias=True)
        self.deconvolution = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        def he_initialization(layer):
            nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(2 / (layer.out_channels * layer.weight[0][0].numel())))
            nn.init.zeros_(layer.bias)

        he_initialization(self.feature_extraction)
        he_initialization(self.shrinking)
        
        for layer in self.mapping:
            if isinstance(layer, nn.Conv2d):
                he_initialization(layer)

        he_initialization(self.expanding)
        
        nn.init.normal_(self.deconvolution.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconvolution.bias)

    def forward(self, x):
        x = self.relu(self.feature_extraction(x))
        x = self.relu(self.shrinking(x))
        x = self.mapping(x)
        x = self.channel_attention(x)
        x = self.relu(self.expanding(x))
        x = self.deconvolution(x)
        return x