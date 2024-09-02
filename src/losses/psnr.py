import torch
import torch.nn as nn

class PSNRLoss(nn.Module):
    def __init__(self, max_pixel_value=1.0):
        super(PSNRLoss, self).__init__()
        self.max_pixel_value = max_pixel_value

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2)
        psnr = 20 * torch.log10(self.max_pixel_value / torch.sqrt(mse))
        return -psnr
    
class ExpPSNRLoss(nn.Module):
    def __init__(self, max_pixel_value=1.0, alpha=0.1):
        super(ExpPSNRLoss, self).__init__()
        self.max_pixel_value = max_pixel_value
        self.alpha = alpha

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2)
        psnr = 20 * torch.log10(self.max_pixel_value / torch.sqrt(mse))
        loss = torch.exp(-self.alpha * psnr)
        return loss
    
class InversePSNRLoss(nn.Module):
    def __init__(self, max_pixel_value=1.0, epsilon=1e-8):
        super(InversePSNRLoss, self).__init__()
        self.max_pixel_value = max_pixel_value
        self.epsilon = epsilon

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2)
        psnr = 20 * torch.log10(self.max_pixel_value / torch.sqrt(mse))
        loss = 1 / (psnr + self.epsilon)
        return loss