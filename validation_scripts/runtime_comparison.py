import torch
import torch.nn as nn


# モデルインスタンスの生成
model = SRCNN(num_channels=3).cuda().half()

# モデルを評価モードに設定
model.eval()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# ダミー入力データの生成 (例: 3チャネルの1024x1024ピクセルの画像をバッチサイズ1で)
input_tensor = torch.rand(1, 3, 1024, 1024).cuda().half()

# アップスケーリング（バイキュービック補間）
upsampled_input = nn.functional.interpolate(input_tensor, scale_factor=4, mode='bicubic', align_corners=False)

start.record()
for _ in range(60):
    # モデルに入力データを通して出力を取得
    upsampled_input = nn.functional.interpolate(input_tensor, scale_factor=4, mode='bicubic', align_corners=False)
    #output_tensor = model(upsampled_input)

end.record()

torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)

print(elapsed_time / 1000, 'sec.')

# 出力のサイズを表示
#print("Output tensor shape:", output_tensor.shape)