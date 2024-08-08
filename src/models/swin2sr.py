import torch
from torch import nn
import torch.nn.functional as F
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

class Swin2SR(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.swin2sr = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
        self.processor = Swin2SRImageProcessor(do_rescale=False, pad_size = 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device_type)
        pixel_values = self.processor(x, return_tensors="pt").pixel_values.to(self.device_type)
        output_image = self.swin2sr(pixel_values=pixel_values).reconstruction
        resized_output = F.interpolate(output_image, size=(x.shape[2] * 4, x.shape[3] * 4), mode='bicubic', align_corners=False)
        return resized_output