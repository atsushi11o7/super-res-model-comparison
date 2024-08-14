import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution


class Swin2SR(nn.Module):
    def __init__(self, model_name="caidas/swin2SR-classical-sr-x4-64"):
        super(Swin2SR, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = next(self.model.parameters()).device
        x = x.to(device_type)
        inputs = self.processor(x, return_tensors="pt").to(device_type)
        outputs = self.model(**inputs).reconstruction
        resized_outputs = F.interpolate(outputs, size=(x.shape[2] * 4, x.shape[3] * 4), mode='bicubic', align_corners=False)
        return resized_outputs