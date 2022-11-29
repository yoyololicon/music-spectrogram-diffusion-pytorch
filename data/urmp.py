import soundfile as sf
from pathlib import Path
import os
import note_seq

from .common import Base


class URMP(Base):
    # MT3 uses same val and test split
    _val_ids = set(["01", "02", "12", "13", "24", "25", "31", "38", "39"])
    _num_tracks_total = 44

    def __init__(
        self,
        wav_path: str = "/import/c4dm-datasets/URMP/Dataset/",
        midi_path: str = "/import/c4dm-datasets/URMP-clean-midi/",
        split: str = "train",
        **kwargs,
    ):
        wav_path = Path(wav_path)
        midi_path = Path(midi_path)
        train_mapping = dict()
        val_mapping = dict()

        for track_id in range(1, self._num_tracks_total + 1):
            track_id = f"{track_id:02d}"

            wav_file = list(
                wav_path.glob(os.path.join(str(track_id) + "_*", "AuMix*.wav"))
            )[0]
            full_name = os.path.basename(os.path.dirname(wav_file))
            midi_file = midi_path / (full_name + ".mid")

            if track_id in self._val_ids:
                val_mapping[track_id] = (wav_file, midi_file)
            else:
                train_mapping[track_id] = (wav_file, midi_file)

        if split == "train":
            mapping = train_mapping
        elif split == "val":
            mapping = val_mapping
        elif split == "test":
            mapping = val_mapping
        else:
            raise ValueError(f"Invalid split: {split}")

        data_list = []
        print("Loading URMP...")
        for track_id, (wav_file, midi_file) in mapping.items():
            info = sf.info(wav_file)
            sr = info.samplerate
            frames = info.frames
            ns = note_seq.midi_file_to_note_sequence(midi_file)
            ns = note_seq.apply_sustain_control_changes(ns)
            data_list.append((wav_file, ns, sr, frames))

        super().__init__(data_list, **kwargs)
