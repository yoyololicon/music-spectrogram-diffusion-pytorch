import jams
import os
from tqdm import tqdm
import note_seq
from note_seq.midi_io import midi_to_note_sequence
import pretty_midi
import soundfile as sf

from .common import Base


def get_noteseq(x: jams.JAMS):
    tmp = [i["data"]
           for i in x["annotations"] if i["namespace"] == "note_midi"]

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)
    for note_list in tmp:
        inst = pretty_midi.Instrument(
            program=25, is_drum=False, name='acoustic guitar (steel)')
        midi_data.instruments.append(inst)
        for note in note_list:
            inst.notes.append(pretty_midi.Note(
                120, round(note[2]), note[0], note[0] + note[1]))
    noteseq = midi_to_note_sequence(midi_data)
    return noteseq


class GuitarSet(Base):
    def __init__(self,
                 path: str = "/import/c4dm-datasets/GuitarSet",
                 split: str = "train",
                 **kwargs):
        data_list = []
        file_names = os.listdir(f"{path}/annotation")
        if split == "train":
            file_names = [
                file for file in file_names if file.split("-")[0][-1] != "3"]
        elif split == "val" or split == "test":
            file_names = [
                file for file in file_names if file.split("-")[0][-1] == "3"]
        else:
            raise ValueError(f'Invalid split: {split}')
        
        for file in tqdm(file_names):
            tmp = jams.load(f"{path}/annotation/{file}")
            title = tmp["file_metadata"]["title"]

            wav_file = f"{path}/audio_mono-mic/{title}_mic.wav"
            info = sf.info(wav_file)
            sr = info.samplerate
            frames = info.frames
            ns = get_noteseq(tmp)
            ns = note_seq.apply_sustain_control_changes(ns)
            data_list.append((wav_file, ns, sr, frames))

        super().__init__(data_list, **kwargs)
