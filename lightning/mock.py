import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.mock import MockSpecDataset, MockAudioDataset


class MockData(pl.LightningDataModule):
    def __init__(self, batch_size: int, use_wav: bool = False, with_context: bool = False):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MockSpecDataset(with_context=self.hparams.with_context) if not self.hparams.use_wav else MockAudioDataset(
                with_context=self.hparams.with_context)
            self.val_dataset = MockSpecDataset(with_context=self.hparams.with_context) if not self.hparams.use_wav else MockAudioDataset(
                with_context=self.hparams.with_context)

        if stage == "test":
            self.test_dataset = MockSpecDataset(with_context=self.hparams.with_context) if not self.hparams.use_wav else MockAudioDataset(
                with_context=self.hparams.with_context)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=1)
