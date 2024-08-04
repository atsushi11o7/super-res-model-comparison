import torch
from torch.nn import MSELoss
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate

class LitModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.model = instantiate(cfg.model)  # Hydra で指定されたモデルをインスタンス化
        self.loss_fn = MSELoss()
        self.learning_rate = cfg.train.learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        low_res, high_res = batch
        preds = self.model(low_res)
        loss = self.loss_fn(preds, high_res)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        low_res, high_res = batch
        preds = self.model(low_res)
        loss = self.loss_fn(preds, high_res)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer