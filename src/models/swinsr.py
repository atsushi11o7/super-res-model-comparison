import torch
import torch.nn as nn
from omegaconf import ListConfig 
from timm.models.swin_transformer import SwinTransformer
from timm.models.swin_transformer_v2 import SwinTransformerV2

class SwinSR(nn.Module):
    def __init__(self, upscale_factor=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super(SwinSR, self).__init__()
        
        # Swin Transformerバックボーン
        self.swin_transformer = SwinTransformer(
            img_size=None,  # サイズを動的に対応させるためNoneに設定
            patch_size=4,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        # アップサンプリング層
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )
        
        # 再構成層
        self.reconstruction = nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # 入力画像のサイズを取得してSwin Transformerに設定
        self.swin_transformer.img_size = x.shape[-2:] 
        features = self.swin_transformer(x)
        upsampled = self.upsample(features)
        out = self.reconstruction(upsampled)
        return out


class Swin2SR(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], upscale_factor=4):
        super(Swin2SR, self).__init__()

        if isinstance(depths, ListConfig):
            depths = list(depths)

        if isinstance(num_heads, ListConfig):
            num_heads = list(num_heads)
        
        # Swin Transformer V2のバックボーン
        self.swin_transformer = SwinTransformerV2(
            img_size=None,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        # アップサンプリング層
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )
        
        # 再構成層
        self.reconstruction = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Swin Transformerで特徴抽出
        features = self.swin_transformer(x)
        
        # アップサンプリング
        upsampled = self.upsample(features)
        
        # 高解像度画像の再構成
        out = self.reconstruction(upsampled)
        return out