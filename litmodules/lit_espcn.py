from pathlib import Path
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
import cv2
import numpy as np

from models import ESPCN
from .utils import calc_psnr, to_onnx

class LitESPCN(LightningModule):
    def __init__(self, learning_rate: float, milestones: list, gamma: float, output_dir: str):
        super().__init__()
        self.model = ESPCN()
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.gamma = gamma
        self.criterion = MSELoss()
        self.output_dir = output_dir

    def forward(self, x):
        return self.model(x)

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