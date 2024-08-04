import torch
import torch.nn as nn


model = SRCNN(num_channels=3).cuda().half()
model.eval()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# ダミー入力
input_tensor = torch.rand(1, 3, 1024, 1024).cuda().half()

start.record()
for _ in range(60):
    output_tensor = model(input_tensor)

end.record()

torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)

print(elapsed_time / 1000, 'sec.')