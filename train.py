import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from .lit_model import LitModel
from .datamodules.datamodule import SRDataModule

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # モデルの初期化
    model = LitModel(cfg)
    
    # データモジュールの初期化
    data_module = SRDataModule(
        train_dir=cfg.paths.train_dir,
        val_dir=cfg.paths.val_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )
    
    # Wandb Loggerの初期化
    wandb_logger = WandbLogger(project=cfg.wandb.project_name)
    
    # トレーナーの初期化と学習開始
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=wandb_logger,
        gpus=1 if cfg.train.use_gpu else 0
    )
    
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()