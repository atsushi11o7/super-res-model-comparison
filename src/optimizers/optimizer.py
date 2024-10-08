import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

def get_optimizer(name, params, **kwargs):
    if name == "Adam":
        return optim.Adam(params, **kwargs)
    elif name == "SGD":
        return optim.SGD(params, **kwargs)
    elif name == "AdamW":
        return optim.AdamW(params, **kwargs)
    elif name == "NAdam":
        return optim.NAdam(params, **kwargs)

    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_scheduler(name, optimizer, **kwargs):
    if name == "StepLR":
        return StepLR(optimizer, **kwargs)

    elif name == "MultiStepLR":
        return MultiStepLR(optimizer, **kwargs)

    else:
        raise ValueError(f"Unsupported scheduler: {name}")