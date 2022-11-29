import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, default_collate
from torch.nn.utils.rnn import pad_sequence
from data.musicnet import MusicNet
from data.maestro import Maestro
from preprocessor.event_codec import Codec


def get_padding_collate_fn(output_size: int):
    def collate_fn(batch):
        """Pad the batch to the longest sequence."""
        seqs = [item[0] for item in batch]
        rest = [item[1:] for item in batch]
        rest = default_collate(rest)
        if output_size is not None:
            seqs = [torch.cat([seq, seq.new_zeros(output_size - len(seq))])
                    for seq in seqs]
            seqs = torch.stack(seqs, dim=0)
        else:
            seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs, *rest
    return collate_fn


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
        collate_fn = get_padding_collate_fn(self.hparams.midi_output_size)
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        collate_fn = get_padding_collate_fn(self.hparams.midi_output_size)
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        collate_fn = get_padding_collate_fn(self.hparams.midi_output_size)
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
