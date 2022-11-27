import soundfile as sf
from pathlib import Path
import numpy as np
import note_seq
from tqdm import tqdm

from .common import Base


class MusicNet(Base):
    _val_ids = ['2336', '2466', '2160', '1818', '1733', '1765', '2198',
                '2300', '2308', '2477', '2611', '2289', '1790', '2315', '2504']
    _test_ids = ['2118', '2501', '1813', '1729', '1893', '2296', '1776', '2487',
                 '2537', '2186', '2431', '2432', '2497', '2621', '2507']

    def __init__(self,
                 path: str,
                 split: str = 'train',
                 **kwargs):

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
            mapping = train_mapping
        elif split == 'val':
            mapping = val_mapping
        elif split == 'test':
            mapping = test_mapping
        else:
            raise ValueError(f'Invalid split: {split}')

        data_list = []
        print("Loading MusicNet...")
        for track_id, (wav_file, midi_file) in tqdm(mapping.items()):
            info = sf.info(wav_file)
            sr = info.samplerate
            frames = info.frames
            ns = note_seq.midi_file_to_note_sequence(midi_file)
            ns = note_seq.apply_sustain_control_changes(ns)
            data_list.append((wav_file, ns, sr, frames))

        super().__init__(data_list, **kwargs)
