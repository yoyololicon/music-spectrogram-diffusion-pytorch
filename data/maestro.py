import torch
import soundfile as sf
from pathlib import Path
import numpy as np
import os
import json

from musicnet import MusicNet


class Maestro(MusicNet):
    def __init__(
        self,
        path: str,
        split: str = "train",
        sample_rate: int = None,
        segment_length: int = 81920,
        with_context: bool = False,
        **midi_kwargs
    ):
        super().__init__(
            self, path, split, sample_rate, segment_length, with_context, **midi_kwargs
        )

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
