import torch
from torch.utils import data
from pathlib import Path
from typing import Tuple, List
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, UnidentifiedImageError
import pytorch_lightning as pl

class DataSetBase(data.Dataset):
    def __init__(self, low_res_dirs: List[Path], high_res_dir: Path):
        self.low_res_images = []
        self.high_res_dir = high_res_dir

        # 各低解像度ディレクトリから画像を収集
        for low_res_dir in low_res_dirs:
            self.low_res_images.extend(list(low_res_dir.iterdir()))

        self.max_num_sample = len(self.low_res_images)
        
    def __len__(self) -> int:
        return self.max_num_sample
    
    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return image  # バリデーションではそのまま使用
    
    def preprocess_low_resolution_image(self, image: Image) -> Image:
        return image  # バリデーションではそのまま使用
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        low_res_image_path = self.low_res_images[index % len(self.low_res_images)]

        low_res_image = Image.open(low_res_image_path)

        high_res_image_path = self.get_corresponding_high_resolution_image(low_res_image_path)
        high_res_image = Image.open(high_res_image_path)
        
        # トレーニングかバリデーションかによって、画像の前処理を分ける
        if isinstance(self, TrainDataSet):
            high_res_image, low_res_image = self.preprocess_images_for_training(high_res_image, low_res_image)
        else:
            low_res_image = self.preprocess_low_resolution_image(low_res_image)
            high_res_image = self.preprocess_high_resolution_image(high_res_image)
        
        return transforms.ToTensor()(low_res_image), transforms.ToTensor()(high_res_image)
    
    def get_corresponding_high_resolution_image(self, low_res_image_path: Path) -> Path:
        if 'x' in low_res_image_path.stem:
            base_name = low_res_image_path.stem.split('x')[0]  # ex): "0001x4" -> "0001"
        else:
            base_name = low_res_image_path.stem

        high_res_image_path = self.high_res_dir / f"{base_name}.png"
        return high_res_image_path

class TrainDataSet(DataSetBase):
    def __init__(self, low_res_dirs: List[Path], high_res_dir: Path, crop_size: Tuple[int, int] = (48, 48), scale_factor: int = 4, num_image_per_epoch: int = 1000):
        super().__init__(low_res_dirs, high_res_dir)
        self.max_num_sample = num_image_per_epoch
        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def preprocess_images_for_training(self, high_res_image: Image.Image, low_res_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # 低解像度画像のサイズを取得
        low_res_width, low_res_height = low_res_image.size
        
        # クロップ位置を決定
        left = torch.randint(0, low_res_width - self.crop_size[0] + 1, (1,)).item()
        top = torch.randint(0, low_res_height - self.crop_size[1] + 1, (1,)).item()
        right = left + self.crop_size[0]
        bottom = top + self.crop_size[1]

        low_res_crop = low_res_image.crop((left, top, right, bottom))
        
        # 高解像度画像のクロップ座標をスケールファクターに応じて計算
        left_hr = left * self.scale_factor
        top_hr = top * self.scale_factor
        right_hr = left_hr + self.crop_size[0] * self.scale_factor
        bottom_hr = top_hr + self.crop_size[1] * self.scale_factor

        high_res_crop = high_res_image.crop((left_hr, top_hr, right_hr, bottom_hr))

        # データ拡張（ランダムな水平および垂直反転）を適用
        if torch.rand(1).item() > 0.5:
            low_res_crop = F.hflip(low_res_crop)
            high_res_crop = F.hflip(high_res_crop)
        
        if torch.rand(1).item() > 0.5:
            low_res_crop = F.vflip(low_res_crop)
            high_res_crop = F.vflip(high_res_crop)

        angles = [0, 90, 180, 270]
        angle = angles[torch.randint(0, len(angles), (1,)).item()]  # 0, 90, 180, 270からランダムに選択
        low_res_crop = F.rotate(low_res_crop, angle)
        high_res_crop = F.rotate(high_res_crop, angle)
        
        return high_res_crop, low_res_crop

class ValidationDataSet(DataSetBase):
    def __init__(self, low_res_dirs: List[Path], high_res_dir: Path):
        super().__init__(low_res_dirs, high_res_dir)
        # Validationでは画像をそのまま使用するため、preprocessメソッドはオーバーライドせずに継承されたものを使用

class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(self, train_low_dirs: List[str], train_high_dir: str, val_low_dirs: List[str], val_high_dir: str, batch_size: int, num_workers: int, crop_size: Tuple[int, int] = (48, 48), scale_factor: int = 4):
        super().__init__()
        self.train_low_dirs = [Path(dir) for dir in train_low_dirs]
        self.train_high_dir = Path(train_high_dir)
        self.val_low_dirs = [Path(dir) for dir in val_low_dirs]
        self.val_high_dir = Path(val_high_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def setup(self, stage=None):
        self.train_dataset = TrainDataSet(self.train_low_dirs, self.train_high_dir, crop_size=self.crop_size, scale_factor=self.scale_factor)
        self.val_dataset = ValidationDataSet(self.val_low_dirs, self.val_high_dir)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)