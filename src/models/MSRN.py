import os
import glob
import math
import torch
import random
import numpy as np
import torch.utils.data as data
from decimal import Decimal
from torch import nn
from datetime import datetime


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    '''
    カーネルサイズが奇数の時に、出力画像の幅高さが入力画像と同じになるように
    調整された畳み込み層を返す関数
    ::math
        output_Height = (height + 2padding - kernel_size)/stride + 1
    Parameters
    -----
    in_channels:
        入力画像のチャンネル数 (ex. RGBなら3チャンネル)
    out_channels:
        出力画像のチャンネル数
    kernel_size:
        畳み込み計算にて使用する正方行列の1辺の長さ
    bias:
        畳み込み結果にバイアスを足すかどうかのフラグ
    '''
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias
    )


class MeanShift(nn.Conv2d):
    '''
    平均値を画像全体に(足すor引く)すよう設定、
    重みに関しても、重みの値のばらつきを統一(標準偏差で割る)した状態で、
    畳み込みを計算する関数(カーネルサイズは1, ストライド1なので、出力サイズは変わらない)
    つまり画像に対して正規化を行うクラス
    '''
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(in_channels=3, out_channels=3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        # print(std)  # tensor([1., 1., 1.])
        # torch.eye 全要素1の対角行列の作成
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)  # view: reshape(3, 3, 1, 1)  3つの3行1列1深さ
        self.weight.data.div_(std.view(3, 1, 1, 1))  # weight / std.reshape(3, 1, 1, 1)
        # print(self.weight.data)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    '''
    畳み込み(チャンネル数拡大) ==> PixelShuffle(画像サイズ拡大) ==> (バッチ正規化) ==> 活性化 ==> nn.sequential定義
    '''
    def __init__(self, conv=default_conv, scale=2, n_feats=64, bn=False, act='relu', bias=True):
        '''
        conv: func
          汎用畳み込み関数
        scale: int
          倍率
        n_feats: int
          画像のチャンネル数
        bn: Boolean
          バッチノーマライゼーションを行うか行わないかのフラグ
        act: str
          活性化関数 (relu | prelu)
        '''
        m = []
        self.scale = scale
        self.n_feats = n_feats
        self.bn = bn
        if (self.scale & (self.scale-1))==0:  # python &はビット演算でscale=2のべき乗かを判定
            for _ in range(int(math.log(self.scale, 2))):  # math.log(scale, 2) ==> log2(scale)
                m.append(conv(n_feats, 4*n_feats, 3, bias))  # 出力は4倍(高さ2倍x幅2倍)の画像量をConvにより出力
                # PixelShuffle(r)
                ## r: 倍率
                ## テンソル内の要素を再整列させる関数
                ##  (-1, C*r^2, H, W) to a tensor of shape (-1, C, H*r, W*r), 
                m.append(nn.PixelShuffle(2))
                if bn:
                    # Applies Batch Normalization over a 4D input 
                    # (a mini-batch of 2D inputs with additional channel dimension) as described in the paper
                    # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>
                    m.append(nn.BatchNorm2d(n_feats))
                if act=='relu':
                    m.append(nn.ReLU(inplace=True))
                elif act=='prelu':
                    # \text{PReLU}(x) = \max(0,x) + a * \min(0,x)
                    m.append(nn.PReLU(num_parameters=n_feats))
        elif scale==3:
            m.append(conv(n_feats, 9*n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: 
                m.append(nn.BatchNorm2d(n_feats))
            if act=='relu': 
                m.append(nn.ReLU(inplace=True))
            elif act=='prelu': 
                m.append(nn.PReLU(num_parameters=n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class MSRB(nn.Module):
    '''
    MultiScaleResidualBlock
    nn.Moduleは全てのニューラルネットワークモデルのBaseクラス  
    '''
    def __init__(self, conv=default_conv, n_feats=64):
        '''
        conv: func
          汎用畳み込み関数
        n_feats: int
          画像のチャンネル数
        '''
        super(MSRB, self).__init__()
        kernel_size_1 = 3
        kernel_size_2 = 5
        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats*4, n_feats, 1, padding=0, stride=1)  # 出力された総チャンネル数をn_featsに圧縮
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))  # [画像枚数, チャンネル(64), Height(96), Width(96)]
        output_5_1 = self.relu(self.conv_5_1(input_1))  # [画像枚数, チャンネル(64), Height(96), Width(96)]
        input_2 = torch.cat([output_3_1, output_5_1], 1) # [画像枚数, チャンネル(128), Height(96), Width(96)] 1次元目(=channel数)方向に結合
        output_3_2 = self.relu(self.conv_3_2(input_2))  # [画像枚数, チャンネル(128), Height(96), Width(96)] 
        output_5_2 = self.relu(self.conv_5_2(input_2))  # [画像枚数, チャンネル(128), Height(96), Width(96)] 
        input_3 = torch.cat([output_3_2, output_5_2], 1)  # [画像枚数, チャンネル(256), Height(96), Width(96)] 1次元目(=channerl数)方向に結合
        output = self.confusion(input_3)  # [画像枚数, チャンネル(64), Height(96), Width(96)] ボトルネック層によるチャンネル方向への圧縮
        output += x  # [画像枚数, チャンネル(64), Height(96), Width(96)] 残差ブロック
        return output
    

class MSRN(nn.Module):
    '''
    nn.Moduleは全てのニューラルネットワークモデルのBaseクラス    
    '''
    def __init__(self, conv=default_conv, n_feats=64, msrb_block_num = 8, scale=4, rgb_range=1):
        '''
        conv: func
          汎用畳み込み関数
        n_feats: int
          画像のチャンネル数
        msrb_block_num: int
          MSRB構造の採用個数
        scale: int
          超解像(スケールアップ)の希望倍率
        rgb_range: int
          使用する画像の画素値の最大値
        '''
        super(MSRN, self).__init__()
        self.rgb_range = rgb_range
        self.scale = scale
        self.n_feats = n_feats
        self.n_blocks = msrb_block_num  # MSRBの数
        self.kernel_size = 3
        # NOTE: nn.Moduleクラスを継承したクラス内でnnから呼ばれたオブジェクトはモジュールリストに登録される
        self.relu = nn.ReLU(inplace=True)  

        # RGB mean/std for DIV2K Dataset
        ## rgb_mean: 画素値範囲(0から1の間)の平均値
        ## rgb_std: 同上の標準偏差
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (2.0, 2.0, 3.0)
        self.sub_mean = MeanShift(self.rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(self.rgb_range, rgb_mean, rgb_std, 1)

        # HEAD module
        # modules_head = [conv(args.n_colors, self.n_feats, self.kernel_size)]
        # イメージ：3色情報を複数次元へConvによりマッピング
        modules_head = [ conv(3, self.n_feats, self.kernel_size) ]

        # Body module
        modules_body = nn.ModuleList()  # 必要な層をリストで渡すことで、層のイテレータを作成してくれる。これによってforward処理を簡単に
        for i in range(self.n_blocks):
            modules_body.append( MSRB(n_feats=self.n_feats) )
        
        # Tail module
        ## 入力は、上で設定した各MSRBのブロックからの各出力 + 最初の入力画像
        ### つまり入力画像の種類は、1ブロックあたりのフィルタの数 * (MSRBのブロック数+最初の入力画像)である
        modules_tail = [
            nn.Conv2d(
                in_channels=self.n_feats*(self.n_blocks+1), 
                out_channels=self.n_feats,
                kernel_size=1,
                padding=0,
                stride=1
            ),
            conv(self.n_feats, self.n_feats, self.kernel_size),
            Upsampler(conv, self.scale, self.n_feats, act=False),
            conv(self.n_feats, 3, self.kernel_size)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)  # 画像xを正規化して、畳み込み(画像サイズ自体は不変)
        x = self.head(x)  # 3channel ==> n_feats チャンネルに増量するため畳み込み(画像サイズ自体は不変)
        res = x
        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSRB_out.append(x)  # 各MSRBブロックで出力されたものを随時追加
        MSRB_out.append(res)

        res = torch.cat(MSRB_out, 1)  # MSRB_out: [出力枚数,  チャネル数, 高さ, 幅]のうち1次元目(チャネル数)方向に結合
        x = self.tail(res)  # 畳み込み(チャンネル数拡大) ==> PixelShuffle(画像サイズ拡大) ==> (バッチ正規化) ==> 活性化 
        x = self.add_mean(x)  # 正規化されていた画像xに平均値を足して、畳み込み(画像サイズ自体は不変)
        return x