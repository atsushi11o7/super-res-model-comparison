from pathlib import Path
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
import cv2
import numpy as np
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
import torch.nn.functional as F

from .utils import calc_psnr, to_onnx

class Swin2SRLitModel(pl.LightningModule):
    def __init__(self, learning_rate: float, milestones: list, gamma: float, output_dir: str):
        super().__init__()
        self.model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
        self.processor = Swin2SRImageProcessor(do_rescale=False, pad_size = 16)
        self.milestones = milestones
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.criterion = MSELoss()
        self.output_dir = output_dir
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = x.to(self.my_device)
        pixel_values = self.processor(x, return_tensors="pt").pixel_values.to(self.my_device)
        output_image = self.model(pixel_values=pixel_values).reconstruction
        resized_output = F.interpolate(output_image, size=(x.shape[2] * 4, x.shape[3] * 4), mode='bicubic', align_corners=False)
        return resized_output

    def training_step(self, batch, batch_idx):
        low_resolution_image, high_resolution_image = batch
        output = self(low_resolution_image)
        loss = self.criterion(output, high_resolution_image)
        psnr = calc_psnr(output, high_resolution_image)
        self.log('train_loss', loss)
        self.log('train_psnr', psnr)
        return loss

    def validation_step(self, batch, batch_idx):
        low_resolution_image, high_resolution_image = batch
        output = self(low_resolution_image)
        loss = self.criterion(output, high_resolution_image)
        psnr = calc_psnr(output, high_resolution_image)
        self.log('val_loss', loss)
        self.log('val_psnr', psnr)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]

    def on_train_end(self):
        model_path = Path(self.output_dir) / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        to_onnx(self.model, self.output_dir)