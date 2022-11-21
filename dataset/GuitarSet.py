import torch
import torchaudio
import jams
import os
import tqdm
import pickle
import numpy as np
from note_seq.protobuf import music_pb2
from note_seq.midi_io import midi_to_note_sequence
import pretty_midi


SR = 44100
MEL_PARA_DICT={"n_mels":128, "center":False, "sample_rate":16000, "n_fft":640, "hop_length":320, "f_min":20.0, "f_max":None}



class MyDataset(torch.utils.data.Dataset): #padding等加在getterm #pad放在init #np_to_torch放在
    def __init__(self, path="/import/c4dm-datasets/GuitarSet"):
        self.path = path
        self.para = MEL_PARA_DICT
        self.data = []
        try:
            file_ = open('./data.pk', 'wb')
            self.data = pickle.load(file_)
            file_.close()
        except:
            file_names = os.listdir(f"{path}/annotation")
            file_ = open('./data.pk', 'wb')
            for file in tqdm.tqdm(file_names[:1]):
                tmp = jams.load(f"{path}/annotation/{file}")
                title = tmp["file_metadata"]["title"]
                duration = tmp["file_metadata"]["duration"]
                num_seq = int(duration * 100 + 511) // 512
                self.data.extend([(title, i) for i in range(num_seq)])
            pickle.dump(self.data, file_)
            file_.close()
    
    def __len__(self): # 返回df的长度
        return len(self.data)
    def __getitem__(self, idx): # 获取第idx+1列的数据
        title, time = self.data[idx]
        audio, sr = torchaudio.load(f"{self.path}/audio_mono-pickup_mix/{title}_mix.wav", frame_offset=int(SR*5.12*time), num_frames=int(SR*5.12))
        if time==0:
            context = torch.zeros((1, int(5.12*16000)))
        else:
            context, sr = torchaudio.load(f"{self.path}/audio_mono-pickup_mix/{title}_mix.wav", frame_offset=int(SR*5.12*(time-1)), num_frames=int(SR*5.12))
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = transform(audio)
        context = transform(context)
        if audio.shape[1] < 5.12 * 16000:
            audio = torch.nn.functional.pad(audio, (0, int(5.12 * 16000 - audio.shape[1])))
        audio = self.audio2mel(audio)
        context = self.audio2mel(context)
        
        tmp = jams.load(f"{self.path}/annotation/{title}.jams")
        tmp = [i["data"] for i in tmp["annotations"] if i["namespace"]=="note_midi"]
                
        noteseq = self.midi2noteseq(tmp)
        return noteseq, context, audio
        # return 0
    
    def audio2mel(self, input, melkwargs=MEL_PARA_DICT, pad=5.12):
        # print(input)
        input = input.squeeze(0)
        if input.shape[0] < pad * 16000:
            input = torch.nn.functional.pad(input, (0, int(pad * 16000 - input.shape[0])))
        MelSpec = torchaudio.transforms.MelSpectrogram(**melkwargs)
        amplitude_to_DB = torchaudio.transforms.AmplitudeToDB("power", 80.0)  # 80.0 is the value of top dB
        mel_spec = MelSpec(input)
        log_mel = amplitude_to_DB(mel_spec)
        return log_mel
    
    def midi2noteseq(self, tmp):
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)
        for idx, note_list in enumerate(tmp):
            exec(f"inst{idx} = pretty_midi.Instrument(program=25, is_drum=False, name='acoustic guitar (steel)')")
            exec(f"midi_data.instruments.append(inst{idx})")
            for note in note_list:
                # print(note) #time duratioin value
                pitch = int(note[2])
                start, end = note[0], note[0] + note[1]
                brend = int((note[2] - pitch) * 4096)
                exec(f"inst{idx}.notes.append(pretty_midi.Note(120, pitch, start, end))")
                exec(f"inst{idx}.pitch_bends.append(pretty_midi.PitchBend(brend, start))")
                exec(f"inst{idx}.pitch_bends.append(pretty_midi.PitchBend(brend, end))")  
        noteseq = midi_to_note_sequence(midi_data)
        return midi_data #noteseq
    

    # def collate_fn(batch):
    #     ### Select all data and label from batch
    #     batch_x = [x for x,y in batch]
    #     batch_y = [y for x,y in batch]
    #     ### Convert batched data and labels to tensors
    #     batch_x = torch.as_tensor(batch_x)
    #     batch_y = torch.as_tensor(batch_y)
    #     return batch_x, batch_y
    #     pass
    

if __name__ == "__main__":
    data_set = MyDataset()
    # print(len(data_set))
    # print(data_set[0][1].shape)
    # for i in range(len(data_set)):
    #     midi, context, audio = data_set[i]
    #     # print(len(midi), context.shape, audio.shape)
        
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1)
    for data in data_loader:
        continue
        break
    a = jams.load("/import/c4dm-datasets/GuitarSet/annotation/00_BN1-129-Eb_comp.jams")
    for tmp in a["annotations"]:
        # print(tmp)
        if tmp["namespace"] == "note_midi":
            print(tmp)
