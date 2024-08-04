from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils import TrainDataSet, ValidationDataSet

class SRDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, batch_size: int = 50, num_workers: int = 4):
        super().__init__()
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = TrainDataSet(self.train_dir)
        self.val_dataset = ValidationDataSet(self.val_dir / 'original', self.val_dir / '0.25x')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)