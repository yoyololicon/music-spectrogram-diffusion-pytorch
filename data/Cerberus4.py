"""
The paper google cited for this dataset is 
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054340
which is a model on source separation and transcription

The previous paper use the dataset Slakh2100 (handelled by others?) and MAPS. 
So this File is mainly focus on the data_loader of MAPS.
The MAPS dataset it cited is not the dataset but another paper that use the dataset
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054340

The true dataset is at here
https://c4dm.eecs.qmul.ac.uk/ycart/a-maps.html
"""
import tqdm
import soundfile as sf
from pathlib import Path
import note_seq
import tqdm

from .common import Base
from preprocessor.event_codec import Codec


class MAPS(Base):  # padding等加在getterm #pad放在init #np_to_torch放在
    def __init__(self,
                 path: str = "/import/c4dm-datasets/MAPS_working/MAPS",
                 **kwargs):
        data_list = []
        print("Loading MAPS...")
        resolution = 100
        segment_length_in_time = 5.12
        codec = Codec(int(segment_length_in_time * resolution + 1))
        for wav_file in tqdm.tqdm(list(Path(path).rglob('*.wav'))):
            info = sf.info(wav_file)
            sr = info.samplerate
            frames = info.frames
            midi_file = str(wav_file)[:-3] + "mid"
            ns = note_seq.midi_file_to_note_sequence(midi_file)
            ns = note_seq.apply_sustain_control_changes(ns)
            data_list.append((wav_file, ns, sr, frames))
        super().__init__(data_list, codec=codec, **kwargs)


if __name__ == "__main__":
    data_set = MAPS()

    # data_loader = torch.utils.data.DataLoader(data_set, batch_size=1)
    # for data in data_loader:
    #     print(data[0], data[1].shape, data[2].shape)
    #     break
