import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datamodule import SuperResolutionDataModule
from litmodules.lit_espcn import LitESPCN
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Setup wandb logger
    wandb_logger = WandbLogger(project=cfg.wandb.project_name)
    
    # Setup data module
    data_module = SuperResolutionDataModule(
        train_dir=cfg.dataset.train_data_path,
        val_high_dir=cfg.dataset.original_val_data_path,
        val_low_dir=cfg.dataset.quarter_val_data_path,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )
    
    # Setup model
    model = LitESPCN(
        learning_rate=cfg.train.learning_rate,
        milestones=cfg.train.milestones,
        gamma=cfg.train.gamma,
        output_dir=cfg.train.output_dir
    )
    
    # Setup trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.train.num_epoch,
        logger=wandb_logger,
        default_root_dir=cfg.train.log_dir,
    )
    
    # Start training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()