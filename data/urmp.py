import torch
import soundfile as sf
from pathlib import Path
import numpy as np
import os

from musicnet import MusicNet

class URMP(MusicNet):
    # MT3 uses same val and test split
    _val_ids = set(['01', '02', '12', '13', '24', '25', '31', '38', '39'])
    _num_tracks_total = 44

    def __init__(self,
                 path: str = '/import/c4dm-datasets/URMP/Dataset/',
                 midi_path: str = '/import/c4dm-datasets/URMP-clean-midi/',
                 split: str = 'train',
                 sample_rate: int = None,
                 segment_length: int = 81920,
                 with_context: bool = False,
                 **midi_kwargs):
        super().__init__()

        path = Path(path)
        midi_path = Path(midi_path)
        train_mapping = dict()
        val_mapping = dict()

        for track_id in range(1, self._num_tracks_total + 1):
            track_id = f"{track_id:02d}"
            
            wav_file = list(path.glob(os.path.join(str(track_id) + '_*', 'AuMix*.wav')))[0]
            full_name = os.path.basename(os.path.dirname(wav_file))
            midi_file = midi_path / (full_name + '.mid')

            if track_id in self._val_ids:
                val_mapping[track_id] = (wav_file, midi_file)
            else:
                train_mapping[track_id] = (wav_file, midi_file)

        if split == 'train':
            self.mapping = train_mapping
        elif split == 'val':
            self.mapping = val_mapping
        elif split == 'test':
            self.mapping = val_mapping
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

