import torch
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path
import resampy
import numpy as np
from typing import Tuple, Union, Optional, List, Dict, Any
from preprocessor.event_codec import Codec


class Base(Dataset):
    def __init__(self,
                 data_list: List[Tuple[Path, Union[np.ndarray, torch.Tensor], int, int]],
                 sample_rate: int = None,
                 segment_length: int = 81920,
                 with_context: bool = False):
        super().__init__()

        boundaries = [0]
        self.data_list = []
        for filename, midi, sr, frames in data_list:
            if sample_rate is None:
                segment_length_in_time = segment_length / sr
            else:
                segment_length_in_time = segment_length / sample_rate
            num_chunks = frames // (segment_length_in_time * sr)
            boundaries.append(boundaries[-1] + num_chunks)
            self.data_list.append((filename, midi, sr, segment_length_in_time))

        self.data_list = data_list
        self.boundaries = np.array(boundaries)
        self.sr = sample_rate
        self.segment_length = segment_length
        self.with_context = with_context

    def __len__(self) -> int:
        return self.boundaries[-1]

    def _get_file_idx_and_chunk_idx(self, index: int) -> Tuple[int, int]:
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        chunk_index = index - self.boundaries[bin_pos]
        return bin_pos, chunk_index

    def _get_waveforms(self, index: int, chunk_index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get waveform without resampling."""
        wav_file, _, sr, length_in_time = self.data_list[index]
        offset = int(chunk_index * length_in_time * sr)
        frames = int(length_in_time * sr)

        if not self.with_context:
            data, _ = sf.read(
                wav_file, start=offset, frames=frames, dtype='float32', always_2d=True)
            data = data.mean(axis=1)
            return data

        ctx_offset = offset - frames
        if ctx_offset >= 0:
            ctx, _ = sf.read(wav_file, start=ctx_offset,
                             frames=frames * 2, dtype='float32', always_2d=True)
            data = ctx[frames:]
            ctx = ctx[:frames]
        else:
            data, _ = sf.read(
                wav_file, start=offset, frames=frames, dtype='float32', always_2d=True)
            ctx = np.zeros_like(data)
        data = data.mean(axis=1)
        ctx = ctx.mean(axis=1)
        return data, ctx

    def __getitem__(self, index: int) -> torch.Tensor:
        file_idx, chunk_idx = self._get_file_idx_and_chunk_idx(index)
        tokens = None

        if not self.with_context:
            data = self._get_waveforms(file_idx, chunk_idx)
            if data.shape[0] != self.segment_length:
                data = resampy.resample(data, data.shape[0], self.segment_length, axis=0, filter='kaiser_fast')[:self.segment_length]
                if data.shape[0] < self.segment_length:
                    data = np.pad(
                        data, ((0, self.segment_length - data.shape[0]),), 'constant')
            return tokens, data

        data, ctx = self._get_waveforms(file_idx, chunk_idx)

        if data.shape[0] != self.segment_length:
            tmp = np.vstack([ctx, data])
            tmp = resampy.resample(tmp, data.shape[0], self.segment_length, axis=1, filter='kaiser_fast')[
                :, :self.segment_length]
            if tmp.shape[1] < self.segment_length:
                tmp = np.pad(
                    tmp, ((0, 0), (0, self.segment_length - tmp.shape[1])), 'constant')
            ctx, data = tmp
        return tokens, data, ctx