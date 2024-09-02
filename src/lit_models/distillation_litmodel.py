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

class LitModelWithDistillation(pl.LightningModule):
    def __init__(
        self,
        model_config,
        loss_config,
        optimizer_config,
        scheduler_config,
        output_dir,
        weights_path,
        ssim_loss_alpha,
        temperature=4.0,
        alpha=0.5
        ):
        super(LitModelWithDistillation, self).__init__()

        # 教師モデルの設定
        teacher_model_params = {k: v for k, v in model_config.items() if k != "name"}
        self.teacher_model = get_model("ESPCNWithResidualBlockTTA", **teacher_model_params)
        
        teacher_weights_path = "models/espcn_rb_tta/base/checkpoints/epoch=48-val_psnr=28.8440.ckpt"
        checkpoint = torch.load(teacher_weights_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('model.', '')
            new_state_dict[name] = v
        self.teacher_model.load_state_dict(new_state_dict)
        self.teacher_model.eval()  # 教師モデルは評価モードで固定
        print(f"Teacher model weights loaded from {teacher_weights_path}")

        # 生徒モデルの設定
        student_model_params = {k: v for k, v in model_config.items() if k != "name"}
        self.model = get_model(model_config.name, **student_model_params)

        # 生徒モデルの重みをロード
        if weights_path:
            checkpoint = torch.load(weights_path)
            state_dict = checkpoint['state_dict']
            #tmp_state_dict = {k.replace('student_', ''): v for k, v in state_dict.items() if k.startswith('student_')}
            #new_state_dict = {}
            #for k, v in tmp_state_dict.items():
            #    name = k.replace('model.', '')
            #    new_state_dict[name] = v
            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            self.model.load_state_dict(new_state_dict)
            print(f"Student model weights loaded from {weights_path}")
        else:
            print("No student weights provided, using student model with random initialization.")

        loss_params = {k: v for k, v in loss_config.items() if k != "name"}
        self.loss_function = get_loss_function(loss_config.name, **loss_params)
        self.ssim_loss_alpha = ssim_loss_alpha

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.output_dir = output_dir

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, x):
        return self.model(x)

    def distillation_loss(self, student_output, teacher_output, high_resolution_image):
        """
        蒸留損失の計算を行います。これは、生徒モデルが教師モデルを模倣することを目的としています。

        Args:
            student_output (torch.Tensor): 生徒モデルの出力。
            teacher_output (torch.Tensor): 教師モデルの出力。
            high_resolution_image (torch.Tensor): 高解像度のターゲット画像。

        Returns:
            torch.Tensor: 合計損失。
        """

        # ピクセル単位の損失（MSE）
        student_loss = self.loss_function(student_output, high_resolution_image)

        # 教師モデルと生徒モデルの出力間のL2距離損失
        distill_loss = F.mse_loss(student_output, teacher_output)

        # 合計損失の計算：ピクセル損失と蒸留損失の重み付け和
        loss = student_loss + 4 * distill_loss

        return loss
    
    def training_step(self, batch, batch_idx):
        low_resolution_image, high_resolution_image = batch

        # 教師モデルの出力
        with torch.no_grad():
            teacher_output = self.teacher_model(low_resolution_image)

        # 生徒モデルの出力
        student_output = self(low_resolution_image)
        
        # 蒸留損失の計算
        loss = self.distillation_loss(student_output, teacher_output, high_resolution_image)

        ssim = self.ssim(student_output, high_resolution_image)
        psnr = calc_psnr(student_output, high_resolution_image)

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
        psnr = calc_psnr(output, high_resolution_image)

        if self.ssim_loss_alpha:
            loss = (1 - self.ssim_loss_alpha) * loss + self.ssim_loss_alpha * (1 - ssim)

        self.log('val_loss', loss)
        self.log('val_ssim', ssim)
        self.log('val_psnr', psnr)

    def configure_optimizers(self):
        optimizer_params = {k: v for k, v in self.optimizer_config.items() if k != "name"}
        optimizer = get_optimizer(self.optimizer_config.name, self.model.parameters(), **optimizer_params)

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
        torch.save(self.student_model.state_dict(), model_path)
        print(f"Student model saved to {model_path}")
        
        to_onnx(self.student_model, output_dir_path) 
    """