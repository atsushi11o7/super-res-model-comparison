import torch.nn as nn

def get_loss_function(name, **kwargs):
    if name == "MSELoss":
        return nn.MSELoss(**kwargs)

    else:
        raise ValueError(f"Unsupported loss function: {name}")