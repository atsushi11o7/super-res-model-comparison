import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.datamodules import BicubicSuperResolutionDataModule, SuperResolutionDataModule
from src.lit_models import LitModelWithDistillation
from pathlib import Path
import hydra
from omegaconf import DictConfig

from src.lit_models.utils import get_model, to_onnx


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        )
    
    if cfg.datamodule.type == "bicubic":
        data_module = BicubicSuperResolutionDataModule(
            train_dir=cfg.datamodule.dataset.train_data_path,
            val_high_dir=cfg.datamodule.dataset.original_val_data_path,
            val_low_dir=cfg.datamodule.dataset.quarter_val_data_path,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers
        )
        print("Using bicubic data module.")

    elif cfg.datamodule.type == "normal":
        data_module = SuperResolutionDataModule(
            train_low_dirs=cfg.datamodule.dataset.train_data_path.lr_dirs,
            train_high_dir=cfg.datamodule.dataset.train_data_path.hr,
            val_low_dirs=cfg.datamodule.dataset.val_data_path.lr_dirs,
            val_high_dir=cfg.datamodule.dataset.val_data_path.hr,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers
        )
        print("Using normal data module.")

    else:
        raise ValueError(f"Unknown datamodule type: {cfg.datamodule.type}")

    # Setup model
    model = LitModelWithDistillation(
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
    checkpoint_callback_loss = ModelCheckpoint(
        monitor='val_loss',  # ex) 'val_psnr', 'val_ssim'
        dirpath=Path(cfg.train.output_dir).joinpath("checkpoints"),
        filename='{epoch:02d}-{val_loss:.8f}',  # ファイル名
        save_top_k=3,  # 最も良いモデルのみ保存
        mode='min',  # 'val_loss' なら 'min', 'val_psnr' や 'val_ssim' なら 'max'
        save_weights_only=True,  # モデルの重みのみを保存
    )

    checkpoint_callback_psnr_train = ModelCheckpoint(
        monitor='train_psnr',  # ex) 'val_psnr', 'val_ssim'
        dirpath=Path(cfg.train.output_dir).joinpath("checkpoints"),
        filename='{epoch:02d}-{train_psnr:.4f}',  # ファイル名
        save_top_k=3,  # 最も良いモデルのみ保存
        mode='max',  # 'val_loss' なら 'min', 'val_psnr' や 'val_ssim' なら 'max'
        save_weights_only=True,  # モデルの重みのみを保存
    )

    checkpoint_callback_psnr = ModelCheckpoint(
        monitor='val_psnr',  # ex) 'val_psnr', 'val_ssim'
        dirpath=Path(cfg.train.output_dir).joinpath("checkpoints"),
        filename='{epoch:02d}-{val_psnr:.4f}',  # ファイル名
        save_top_k=3,  # 最も良いモデルのみ保存
        mode='max',  # 'val_loss' なら 'min', 'val_psnr' や 'val_ssim' なら 'max'
        save_weights_only=True,  # モデルの重みのみを保存
    )

    # Setup trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback_loss, checkpoint_callback_psnr_train, checkpoint_callback_psnr],
        max_epochs=cfg.train.num_epoch,
        logger=wandb_logger,
        default_root_dir=cfg.train.log_dir
    )
    
    # Start training
    trainer.fit(model, data_module)

    model_params = {k: v for k, v in cfg.model.items() if k != "name"}
    save_model = get_model(cfg.model.name, **model_params)

    best_model_path = checkpoint_callback_psnr.best_model_path
    if best_model_path:
        checkpoint = torch.load(best_model_path)
        state_dict = checkpoint['state_dict']
        #new_state_dict = {}
        #for k, v in state_dict.items():
            #name = k.replace('model.', '')
            #new_state_dict[name] = v
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        save_model.load_state_dict(new_state_dict)
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