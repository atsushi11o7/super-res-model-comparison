import torch.nn as nn
from src.losses.psnr import ExpPSNRLoss


def get_loss_function(name, **kwargs):
    if name == "MSELoss":
        return nn.MSELoss(**kwargs)
    elif name == "L1Loss":
        return nn.L1Loss(**kwargs)
    elif name == "ExpPSNRLoss":
        return ExpPSNRLoss(**kwargs)

    else:
        raise ValueError(f"Unsupported loss function: {name}")