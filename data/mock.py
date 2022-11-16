import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple


class MockSpecDataset(Dataset):
    def __init__(self,
                 freq_bins: int = 128,
                 time_bins: int = 256,
                 num_tokens: int = 2048,
                 num_classes: int = 900,
                 num_chunks: int = 100):

        self.spec_data = torch.rand(num_chunks, time_bins, freq_bins) * 2 - 1
        self.midi_data = torch.randint(
            0, num_classes, (num_chunks, num_tokens))

    def __len__(self) -> int:
        return self.spec_data.shape[0]

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        spec = self.spec_data[idx]
        if idx:
            context = self.spec_data[idx - 1]
        else:
            context = torch.zeros_like(spec)
        midi = self.midi_data[idx]
        return midi, spec, context


class MockAudioDataset(Dataset):
    def __init__(self,
                 num_chunks: int = 100,
                 chunk_size: int = 16000,
                 num_classes: int = 900,
                 num_tokens: int = 2048):

        self.audio_data = torch.rand(num_chunks, chunk_size) * 2 - 1
        self.midi_data = torch.randint(
            0, num_classes, (num_chunks, num_tokens))

    def __len__(self) -> int:
        return self.audio_data.shape[0]

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        audio = self.audio_data[idx]
        if idx:
            context = self.audio_data[idx - 1]
        else:
            context = torch.zeros_like(audio)
        midi = self.midi_data[idx]
        return midi, audio, context
