import torch
import torchaudio
import jams
import os
import tqdm
import note_seq
from note_seq.midi_io import midi_to_note_sequence
import pretty_midi

from .common import Base
from preprocessor.event_codec import Codec


SR = 44100


def get_noteseq(title):
    path = "/import/c4dm-datasets/GuitarSet"
    tmp = jams.load(f"{path}/annotation/{title}.jams")
    tmp = [i["data"]
           for i in tmp["annotations"] if i["namespace"] == "note_midi"]

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)
    for note_list in tmp:
        inst = pretty_midi.Instrument(
            program=25, is_drum=False, name='acoustic guitar (steel)')
        midi_data.instruments.append(inst)
        for note in note_list:
            inst.notes.append(pretty_midi.Note(
                120, int(note[2]), note[0], note[0] + note[1]))
    noteseq = midi_to_note_sequence(midi_data)
    return noteseq


class GuitarSet(Base):  # padding等加在getterm #pad放在init #np_to_torch放在
    def __init__(self,
                 path: str = "/import/c4dm-datasets/GuitarSet",
                 **kwargs):
        data_list = []
        file_names = os.listdir(f"{path}/annotation")
        for file in tqdm.tqdm(file_names):
            tmp = jams.load(f"{path}/annotation/{file}")
            title = tmp["file_metadata"]["title"]
            duration = tmp["file_metadata"]["duration"]

            frames = duration * SR
            wav_file = f"{path}/audio_mono-pickup_mix/{title}_mix.wav"
            y, _ = torchaudio.load(wav_file)
            # ns = note_seq.midi_file_to_note_sequence(midi_file)
            ns = get_noteseq(title)
            ns = note_seq.apply_sustain_control_changes(ns)
            data_list.append((wav_file, ns, SR, frames))

        resolution = 100
        segment_length_in_time = 5.12
        codec = Codec(int(segment_length_in_time * resolution + 1))

        super().__init__(data_list, codec=codec, **kwargs)


if __name__ == "__main__":
    data_set = GuitarSet()

    # data_loader = torch.utils.data.DataLoader(data_set, batch_size=1)
    # for data in data_loader:
    #     continue
    # a = jams.load(
    #     "/import/c4dm-datasets/GuitarSet/annotation/00_BN1-129-Eb_comp.jams")
