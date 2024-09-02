import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio

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
        output_dir,
        weights_path,
        ssim_loss_alpha,
        ):
        super(LitModel, self).__init__()

        model_params = {k: v for k, v in model_config.items() if k != "name"}
        self.model = get_model(model_config.name, **model_params)
        self.tta_model = get_model(model_config.name, **model_params)
        
        if weights_path:
            checkpoint = torch.load(weights_path)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('model.', '')
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

            print(f"Model weights loaded from {weights_path}")
        else:
            print("No weights provided, using model with random initialization.")

        """
        for param in self.model.conv_1.parameters():
            param.requires_grad = False

        for param in self.model.conv_2.parameters():
            param.requires_grad = False

        #for param in self.model.res_block.parameters():
            #param.requires_grad = False

        # Calculate the number of trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Number of trainable parameters: {trainable_params} out of {total_params}")
        """

        loss_params = {k: v for k, v in loss_config.items() if k != "name"}
        self.loss_function = get_loss_function(loss_config.name, **loss_params)
        self.ssim_loss_alpha = ssim_loss_alpha

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.output_dir = output_dir

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        # self.psnr = PeakSignalNoiseRatio(data_range=1.0)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        low_resolution_image, high_resolution_image = batch

        output = self(low_resolution_image)
        loss = self.loss_function(output, high_resolution_image)
        
        ssim = self.ssim(output, high_resolution_image)

        # psnr = self.psnr(output, high_resolution_image)
        psnr =calc_psnr(output, high_resolution_image)

        if self.ssim_loss_alpha:
            loss = (1 - self.ssim_loss_alpha) * loss + self.ssim_loss_alpha * (1 - ssim)

        self.log('train_loss', loss)
        self.log('train_ssim', ssim)
        self.log('train_psnr', psnr)
        return loss


    def validation_step(self, batch, batch_idx):
        low_resolution_image, high_resolution_image = batch
        
        output = self(low_resolution_image)
        loss = self.loss_function(output, high_resolution_image)
        
        ssim = self.ssim(output, high_resolution_image)

        # psnr = self.psnr(output, high_resolution_image)
        psnr = calc_psnr(output, high_resolution_image)

        if self.ssim_loss_alpha:
            loss = (1 - self.ssim_loss_alpha) * loss + self.ssim_loss_alpha * (1 - ssim)

        self.log('val_loss', loss)
        self.log('val_ssim', ssim)
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

    """
    def on_train_end(self):
        output_dir_path = Path(self.output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory {output_dir_path} created.")

        model_path = output_dir_path / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        to_onnx(self.model, output_dir_path) 
    """
