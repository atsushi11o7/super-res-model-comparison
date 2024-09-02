import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodules.datamodule import SuperResolutionDataModule
from src.lit_models import LitModel
from pathlib import Path
import hydra
from omegaconf import DictConfig


from src.lit_models.utils import get_model, to_onnx


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

    # ModelCheckpoint コールバックの定義
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',  # ex) 'val_psnr', 'val_ssim'
        dirpath=cfg.train.output_dir,
        filename='best_checkpoint',  # ファイル名
        save_top_k=1,  # 最も良いモデルのみ保存
        mode='max',  # 'val_loss' なら 'min', 'val_psnr' や 'val_ssim' なら 'max'
        save_weights_only=True,  # モデルの重みのみを保存
    )

    # Setup trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.train.num_epoch,
        logger=wandb_logger,
        default_root_dir=cfg.train.log_dir
    )
    
    # Start training
    trainer.fit(model, data_module)

    model_params = {k: v for k, v in cfg.model.items() if k != "name"}
    save_model = get_model(cfg.model.name, **model_params)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        save_model.load_state_dict(torch.load(best_model_path))
        print(f"Model weights loaded from {best_model_path}")

        output_dir_path = Path(cfg.train.output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory {output_dir_path} created")

        to_onnx(save_model, output_dir_path)
    else:
        print("The best model weights were not found")

if __name__ == "__main__":
    main()