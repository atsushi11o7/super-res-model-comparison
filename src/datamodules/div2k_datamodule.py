import torch
from torch.utils import data
from pathlib import Path
from typing import Tuple, List
from torchvision import transforms
from PIL import Image
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
        return image
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        low_res_image_path = self.low_res_images[index % len(self.low_res_images)]
        low_res_image = Image.open(low_res_image_path)

        high_res_image_path = self.get_corresponding_high_resolution_image(low_res_image_path)
        high_res_image = self.preprocess_high_resolution_image(Image.open(high_res_image_path))
        
        return transforms.ToTensor()(low_res_image), transforms.ToTensor()(high_res_image)
    
    def get_corresponding_high_resolution_image(self, low_res_image_path: Path) -> Path:
        base_name = low_res_image_path.stem.split('x')[0]  # 例: "0001x4.png" -> "0001"
        high_res_image_path = self.high_res_dir / f"{base_name}.png"
        return high_res_image_path

class TrainDataSet(DataSetBase):
    def __init__(self, low_res_dirs: List[Path], high_res_dir: Path, num_image_per_epoch: int = 5000):
        super().__init__(low_res_dirs, high_res_dir)
        self.max_num_sample = num_image_per_epoch

    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose([
            transforms.RandomCrop(size=512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])(image)

class ValidationDataSet(DataSetBase):
    def __init__(self, low_res_dirs: List[Path], high_res_dir: Path):
        super().__init__(low_res_dirs, high_res_dir)

class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(self, train_low_dirs: List[str], train_high_dir: str, val_low_dirs: List[str], val_high_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.train_low_dirs = [Path(dir) for dir in train_low_dirs]
        self.train_high_dir = Path(train_high_dir)
        self.val_low_dirs = [Path(dir) for dir in val_low_dirs]
        self.val_high_dir = Path(val_high_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = TrainDataSet(self.train_low_dirs, self.train_high_dir)
        self.val_dataset = ValidationDataSet(self.val_low_dirs, self.val_high_dir)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)