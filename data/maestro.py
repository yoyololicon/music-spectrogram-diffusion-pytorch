import torch
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path
import numpy as np
import resampy
import json


class Maestro(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        sample_rate: int = None,
        segment_length: int = 81920,
        with_context: bool = False,
        **midi_kwargs
    ):
        super().__init__()

        path = Path(path)
        meta_file = path / "maestrov-v3.0.0.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)
        track_ids = [k for k, v in meta["split"].items() if v == split]
        self.mapping = {
            t: (meta["audio_filename"][t], meta["midi_filename"][t]) for t in track_ids
        }

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
                    data = np.pad(
                        data, ((0, self.segment_length - data.shape[0]),), 'constant')
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
