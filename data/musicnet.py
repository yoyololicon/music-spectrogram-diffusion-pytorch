import torch
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path
import resampy
import numpy as np


class MusicNet(Dataset):
    _val_ids = ['2336', '2466', '2160', '1818', '1733', '1765', '2198',
                '2300', '2308', '2477', '2611', '2289', '1790', '2315', '2504']
    _test_ids = ['2118', '2501', '1813', '1729', '1893', '2296', '1776', '2487',
                 '2537', '2186', '2431', '2432', '2497', '2621', '2507']
    wave_path: str
    midi_path: str

    def __init__(self,
                 path: str,
                 split: str = 'train',
                 sample_rate: int = None,
                 segment_length: int = 81920,
                 with_context: bool = False,
                 **midi_kwargs):
        super().__init__()

        path = Path(path)
        train_mapping = dict()

        for wav_path, midi_path in [
            (path / 'train_data', path / 'train_labels_midi'),
            (path / 'test_data', path / 'test_labels_midi')
        ]:
            for wav_file in wav_path.glob('*.wav'):
                track_id = wav_file.stem
                train_mapping[track_id] = (
                    wav_file, midi_path / f'{track_id}.mid')

        val_mapping = {track_id: train_mapping[track_id]
                       for track_id in self._val_ids}
        test_mapping = {
            track_id: train_mapping[track_id] for track_id in self._test_ids}
        for track_id in self._val_ids:
            del train_mapping[track_id]
        for track_id in self._test_ids:
            del train_mapping[track_id]

        if split == 'train':
            self.mapping = train_mapping
        elif split == 'val':
            self.mapping = val_mapping
        elif split == 'test':
            self.mapping = test_mapping
        else:
            raise ValueError(f'Invalid split: {split}')

        data_list = []
        for track_id, (wav_file, midi_file) in self.mapping.items():
            info = sf.info(wav_file)
            sr = info.samplerate
            if sample_rate is None:
                chunk_length = segment_length
            else:
                chunk_length = int(segment_length * sr / sample_rate)
            chunks = info.frames // chunk_length

            # do something with midi file
            # ...
            midi = None
            data_list.append((wav_file, midi, sr, chunks))

        self.data_list = data_list
        self.boundaries = np.cumsum(np.array([0] + [x[3] for x in data_list]))
        self.sr = sample_rate
        self.segment_length = segment_length
        self.with_context = with_context

    def __getitem__(self, index: int) -> torch.Tensor:
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        wav_file, midi, sr, chunks = self.data_list[bin_pos]
        # tokens = midi[chunk_index]
        tokens = np.random.randint(0, 10, 2048)

        if self.sr is None or sr == self.sr:
            chunk_frames = self.segment_length
        else:
            chunk_frames = int(self.segment_length * sr / self.sr)

        chunk_index = index - self.boundaries[bin_pos]
        offset = chunk_index * chunk_frames

        if not self.with_context:
            data, _ = sf.read(
                wav_file, start=offset, frames=chunk_frames, dtype='float32', always_2d=True)
            data = data.mean(axis=1)
            if self.sr is not None and sr != self.sr:
                data = resampy.resample(data, sr, self.sr, axis=0, filter='kaiser_fast')[
                    :self.segment_length]
                if data.shape[0] < self.segment_length:
                    data = np.pad(data, ((0, self.segment_length - data.shape[0]),), 'constant')
            return tokens, data

        ctx_offset = offset - chunk_frames
        if ctx_offset >= 0:
            ctx, _ = sf.read(wav_file, start=ctx_offset,
                             frames=chunk_frames * 2, dtype='float32', always_2d=True)
            data = ctx[chunk_frames:]
            ctx = ctx[:chunk_frames]
        else:
            data, _ = sf.read(
                wav_file, start=offset, frames=chunk_frames, dtype='float32', always_2d=True)
            ctx = np.zeros_like(data)
        data = data.mean(axis=1)
        ctx = ctx.mean(axis=1)
        if self.sr is not None and sr != self.sr:
            tmp = np.vstack([ctx, data])
            tmp = resampy.resample(tmp, sr, self.sr, axis=1, filter='kaiser_fast')[
                :, :self.segment_length]
            if tmp.shape[1] < self.segment_length:
                tmp = np.pad(
                    tmp, ((0, 0), (0, self.segment_length - tmp.shape[1])), 'constant')
            ctx, data = tmp
        return tokens, data, ctx

    def __len__(self) -> int:
        return self.boundaries[-1]
