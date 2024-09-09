import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from src.models import (
    ESPCN,
    ESPCNWithResidualBlock,
    ESPCNWithResidualBlockV2,
    ESPCNWithResBlockV2AndAttention2,
    ESPCNWithResBlockAndAttention,
    ESPCNWithSimpleResBlock,
    ESPCNWithDense,
    ESPCNWithDenseAndChannelAttention,
    ESPCNWithPixelShuffle2,
    ESPCN2x2,
    EnhancedESPCN,
    ESPCNWithResidualBlockTTA,
    SRCNN,
    FSRCNN,
    FSRCNNWithPixelShuffle,
    FSRCNNWithAttention,
    CARN,
    IMDN,
    HAN,
    IDN,
    VDSR,
    SwinSR,
    Swin2SR,
)


def get_model(name, **kwargs):
    if name == "ESPCN":
        return ESPCN(**kwargs)

    elif name == "ESPCNWithResidualBlock":
        return ESPCNWithResidualBlock(**kwargs)
    
    elif name == "ESPCNWithResidualBlockV2":
        return ESPCNWithResidualBlockV2(**kwargs)
    
    elif name == "ESPCNWithResBlockV2AndAttention2":
        return ESPCNWithResBlockV2AndAttention2(**kwargs)
    
    elif name == "ESPCNWithResBlockAndAttention":
        return ESPCNWithResBlockAndAttention(**kwargs)
    
    elif name == "ESPCNWithSimpleResBlock":
        return ESPCNWithSimpleResBlock(**kwargs)

    elif name == "ESPCNWithDense":
        return ESPCNWithDense(**kwargs)

    elif name == "ESPCNWithDenseAndChannelAttention":
        return ESPCNWithDenseAndChannelAttention(**kwargs)
    
    elif name == "ESPCNWithPixelShuffle2":
        return ESPCNWithPixelShuffle2(**kwargs)
    
    elif name == "ESPCN2x2":
        return ESPCN2x2(**kwargs)
    
    elif name == "ESPCNWithResidualBlockTTA":
        return ESPCNWithResidualBlockTTA(**kwargs)
    
    elif name == "EnhancedESPCN":
        return EnhancedESPCN(**kwargs)
    
    elif name == "SRCNN":
        return SRCNN(**kwargs)

    elif name == "FSRCNN":
        return FSRCNN(**kwargs)
    
    elif name == "FSRCNNWithPixelShuffle":
        return FSRCNNWithPixelShuffle(**kwargs)

    elif name == "FSRCNNWithAttention":
        return FSRCNNWithAttention(**kwargs)

    elif name == "CARN":
        return CARN(**kwargs)

    elif name == "IMDN":
        return IMDN(**kwargs)

    elif name == "HAN":
        return HAN(**kwargs)

    elif name == "IDN":
        return IDN(**kwargs)

    elif name == "VDSR":
        return VDSR(**kwargs)

    elif name == "SwinSR":
        return SwinSR(**kwargs)

    elif name == "Swin2SR":
        return Swin2SR(**kwargs)

    else:
        raise ValueError(f"Unsupported model: {name}")

def calc_psnr(output: torch.Tensor, high_resolution_image: torch.Tensor):
    psnr_transform = transforms.ToPILImage()
    total_psnr = 0
    for image1, image2 in zip(output, high_resolution_image):
        image1 = cv2.cvtColor((np.array(psnr_transform(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor((np.array(psnr_transform(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        total_psnr += cv2.PSNR(image1, image2)
    return total_psnr / len(output)

def to_onnx(model, output_dir):
    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                        opset_version=17,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {2: "height", 3:"width"}})
    print(f"Model exported to ONNX format at {onnx_path}")
