import unittest
from src.div2k_datamodule import SuperResolutionDataModule

class TestSuperResolutionDataModule(unittest.TestCase):
    def test_dataloader(self):
        train_low_dirs = ["data/DIV2K/DIV2K_train_LR_bicubic/X4"]
        train_high_dir = "data/DIV2K/DIV2K_train_HR"
        val_low_dirs = ["data/DIV2K/DIV2K_valid_LR_bicubic/X4"]
        val_high_dir = "data/DIV2K/DIV2K_valid_HR"

        data_module = SuperResolutionDataModule(
            train_low_dirs, train_high_dir,
            val_low_dirs, val_high_dir,
            batch_size=10, num_workers=0
        )
        
        data_module.setup()

        train_loader = data_module.train_dataloader()
        for batch in train_loader:
            low_res, high_res = batch
            self.assertEqual(low_res.shape[0], 10)
            self.assertEqual(high_res.shape[0], 10)
            break

if __name__ == "__main__":
    unittest.main()