import torch
from torch import nn, tensor, Tensor
import onnxruntime as ort
import numpy as np
import datetime
from pathlib import Path
import cv2

class Bicubic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor=4

    def forward(self, x: tensor) -> tensor:
        x = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False
        )
        return x
    
def onnx_model_generation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Bicubic()
    model.to(device)

    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    torch.onnx.export(model, dummy_input, "../onnx_models/model.onnx", 
                    opset_version=17,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {2: "height", 3:"width"}})

def inference_onnxruntime():
    input_image_dir = Path("../data/validation/0.25x")

    sess = ort.InferenceSession("../onnx_models/model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_images = []

    print("load image")
    for image_path in input_image_dir.iterdir():
        input_image = cv2.imread(str(image_path))
        input_image = np.array([cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).transpose((2,0,1))], dtype=np.float32)/255
        input_images.append(input_image)

    print("inference")
    start_time = datetime.datetime.now()
    for input_image in input_images:
        sess.run(None, {"input": input_image})[0]
    end_time = datetime.datetime.now()

    print(f"inference time: {(end_time - start_time).total_seconds() / len(input_images)}[s/image]")

if __name__ == "__main__":
    onnx_model_generation()
    inference_onnxruntime()
