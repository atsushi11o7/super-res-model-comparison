import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from src.lit_models.utils import get_model, calc_psnr, to_onnx
from src.losses.loss import get_loss_function
from src.optimizers.optimizer import get_optimizer, get_scheduler


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model_config,
        loss_config,
        optimizer_config,
        scheduler_config,
        output_dir):
        super(LitModel, self).__init__()

        model_params = {k: v for k, v in model_config.items() if k != "name"}
        self.model = get_model(model_config.name, **model_params)

        loss_params = {k: v for k, v in loss_config.items() if k != "name"}
        self.loss_function = get_loss_function(loss_config.name, **loss_params)

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.output_dir = output_dir


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        low_resolution_image, high_resolution_image = batch
        output = self(low_resolution_image)
        loss = self.loss_function(output, high_resolution_image)
        psnr = calc_psnr(output, high_resolution_image)
        self.log('train_loss', loss)
        self.log('train_psnr', psnr)
        return loss


    def validation_step(self, batch, batch_idx):
        low_resolution_image, high_resolution_image = batch
        output = self(low_resolution_image)
        loss = self.loss_function(output, high_resolution_image)
        psnr = calc_psnr(output, high_resolution_image)
        self.log('val_loss', loss)
        self.log('val_psnr', psnr)


    def configure_optimizers(self):
        optimizer_params = {k: v for k, v in self.optimizer_config.items() if k != "name"}
        optimizer = get_optimizer(self.optimizer_config.name, self.parameters(), **optimizer_params)

        scheduler = None
        if self.scheduler_config.name:
            scheduler_params = {k: v for k, v in self.scheduler_config.items() if k != "name"}
            scheduler = get_scheduler(self.scheduler_config.name, optimizer, **scheduler_params)
        
        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]


    def on_train_end(self):
        model_path = Path(self.output_dir) / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        to_onnx(self.model, self.output_dir) 
