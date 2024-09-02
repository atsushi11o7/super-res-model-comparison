import torch
from torch.utils import data
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

class DataSetBase(data.Dataset):
    def __init__(self, image_path: Path):
        self.images = list(image_path.iterdir())
        self.max_num_sample = len(self.images)
        
    def __len__(self) -> int:
        return self.max_num_sample
    
    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return image
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index % len(self.images)]
        high_resolution_image = self.preprocess_high_resolution_image(Image.open(image_path))
        low_resolution_image = self.get_low_resolution_image(high_resolution_image, image_path)
        return transforms.ToTensor()(low_resolution_image), transforms.ToTensor()(high_resolution_image)
    
    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        raise NotImplementedError

class TrainDataSet(DataSetBase):
    def __init__(self, image_paths: list[Path], num_image_per_epoch: int = 1000):
        self.images = []
        for path in image_paths:
            self.images.extend(list(path.iterdir()))
        self.max_num_sample = num_image_per_epoch

    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        return transforms.Resize((image.size[0] // 4, image.size[1] // 4), transforms.InterpolationMode.BICUBIC)(image.copy())
    
    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose([
            transforms.RandomCrop(size = 192),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation(0),
                transforms.RandomRotation(90),
                transforms.RandomRotation(180),
                transforms.RandomRotation(270),
            ])
        ])(image)

class ValidationDataSet(DataSetBase):
    def __init__(self, high_resolution_image_path: Path, low_resolution_image_path: Path):
        super().__init__(high_resolution_image_path)
        self.high_resolution_image_path = high_resolution_image_path
        self.low_resolution_image_path = low_resolution_image_path

    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        return Image.open(self.low_resolution_image_path / path.relative_to(self.high_resolution_image_path))

class BicubicSuperResolutionDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str, val_high_dir: str, val_low_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.train_dir = [Path(dir) for dir in train_dir]
        self.val_high_dir = Path(val_high_dir)
        self.val_low_dir = Path(val_low_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = TrainDataSet(self.train_dir)
        self.val_dataset = ValidationDataSet(self.val_high_dir, self.val_low_dir)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)