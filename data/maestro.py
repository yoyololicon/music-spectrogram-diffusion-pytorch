import soundfile as sf
from pathlib import Path
import json
import note_seq
from tqdm import tqdm

from .common import Base


class Maestro(Base):
    def __init__(
        self,
        path: str,
        split: str = "train",
        **kwargs
    ):
        path = Path(path)
        meta_file = path / "maestro-v3.0.0.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)
        track_ids = [k for k, v in meta["split"].items() if v == split]
        mapping = {
            t: (path / meta["audio_filename"][t], path / meta["midi_filename"][t]) for t in track_ids
        }

        data_list = []
        print("Loading Maestro...")
        for track_id, (wav_file, midi_file) in tqdm(mapping.items()):
            info = sf.info(wav_file)
            sr = info.samplerate
            frames = info.frames
            ns = note_seq.midi_file_to_note_sequence(midi_file)
            ns = note_seq.apply_sustain_control_changes(ns)
            data_list.append((wav_file, ns, sr, frames))

        super().__init__(data_list, **kwargs)
