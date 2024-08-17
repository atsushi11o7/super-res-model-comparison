import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.datamodules.div2k_datamodule import SuperResolutionDataModule
from src.lit_models import LitModel
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../config", config_name="div2k_config", version_base=None)
def main(cfg: DictConfig):

    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
    )
    
    # Setup data module
    data_module = SuperResolutionDataModule(
        train_low_dirs=cfg.dataset.train_data_path.lr_dirs,
        train_high_dir=cfg.dataset.train_data_path.hr,
        val_low_dirs=cfg.dataset.val_data_path.lr_dirs,
        val_high_dir=cfg.dataset.val_data_path.hr,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )

    # Setup model
    model = LitModel(
        model_config=cfg.model,
        loss_config=cfg.loss,
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        output_dir=cfg.train.output_dir,
        weights_path=cfg.train.weights_path,
        ssim_loss_alpha=cfg.train.ssim_loss_alpha,
    )
    
    torch.cuda.empty_cache()

    # Setup trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.train.num_epoch,
        logger=wandb_logger,
        default_root_dir=cfg.train.log_dir
    )
    
    # Start training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()