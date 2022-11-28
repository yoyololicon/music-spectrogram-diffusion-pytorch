import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from data.musicnet import MusicNet
from data.maestro import Maestro
from preprocessor.event_codec import Codec


class ConcatData(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 midi_output_size: int = 2048,
                 with_context: bool = False,
                 sample_rate: int = 16000,
                 segment_length: int = 81920,
                 musicnet_path: str = None,
                 maestro_path: str = None,
                 slakh_path: str = None,
                 cerberus_path: str = None,
                 guitarset_path: str = None,
                 urmp_path: str = None,
                 ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        resolution = 100
        segment_length_in_time = self.hparams.segment_length / self.hparams.sample_rate
        codec = Codec(int(segment_length_in_time * resolution + 1))

        factory_kwargs = {
            'codec': codec,
            'resolution': resolution,
            'output_size': self.hparams.midi_output_size,
            'sample_rate': self.hparams.sample_rate,
            'segment_length': self.hparams.segment_length,
            'with_context': self.hparams.with_context,
        }

        if stage == "fit" or stage is None:
            train_datasets = []
            val_datasets = []
            if self.hparams.musicnet_path is not None:
                train_datasets.append(
                    MusicNet(path=self.hparams.musicnet_path, split='train', **factory_kwargs))
                val_datasets.append(
                    MusicNet(path=self.hparams.musicnet_path, split='val', **factory_kwargs))

            if self.hparams.maestro_path is not None:
                train_datasets.append(
                    Maestro(path=self.hparams.maestro_path, split='train', **factory_kwargs))
                val_datasets.append(
                    Maestro(path=self.hparams.maestro_path, split='val', **factory_kwargs))

            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)

        if stage == "test":
            test_datasets = []
            if self.hparams.musicnet_path is not None:
                test_datasets.append(
                    MusicNet(path=self.hparams.musicnet_path, split='test', **factory_kwargs))

            if self.hparams.maestro_path is not None:
                test_datasets.append(
                    Maestro(path=self.hparams.maestro_path, split='test', **factory_kwargs))

            self.test_dataset = ConcatDataset(test_datasets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)
