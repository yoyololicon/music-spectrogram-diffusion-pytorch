import torch
import torchaudio
import jams
import os
import tqdm
import pickle
from note_seq.midi_io import midi_to_note_sequence
import pretty_midi


SR = 44100
MEL_PARA_DICT={"n_mels":128, "center":False, "sample_rate":16000, "n_fft":640, "hop_length":320, "f_min":20.0, "f_max":None}



def get_noteseq(title):
    path = "/import/c4dm-datasets/GuitarSet"
    tmp = jams.load(f"{path}/annotation/{title}.jams")
    tmp = [i["data"] for i in tmp["annotations"] if i["namespace"]=="note_midi"]
            
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=25, is_drum=False, name='acoustic guitar (steel)')
    midi_data.instruments.append(inst)
    for note_list in tmp:
        for note in note_list:
            # print(note) #time duratioin value
            pitch = int(note[2])
            start, end = note[0], note[0] + note[1]
            inst.notes.append(pretty_midi.Note(120, pitch, start, end))
            noteseq = midi_to_note_sequence(midi_data)
    return noteseq
    

class MyDataset(torch.utils.data.Dataset): #padding等加在getterm #pad放在init #np_to_torch放在
    def __init__(self, path="/import/c4dm-datasets/GuitarSet"):
        self.path = path
        self.data = []
        try:
            file_ = open('./data.pk', 'wb')
            self.data = pickle.load(file_)
            file_.close()
        except:
            file_names = os.listdir(f"{path}/annotation")
            file_ = open('./data.pk', 'wb')
            for file in tqdm.tqdm(file_names):
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
        return title, context, audio
    
    def audio2mel(self, input, melkwargs=MEL_PARA_DICT, pad=5.12):
        input = input.squeeze(0)
        if input.shape[0] < pad * 16000:
            input = torch.nn.functional.pad(input, (0, int(pad * 16000 - input.shape[0])))
        MelSpec = torchaudio.transforms.MelSpectrogram(**melkwargs)
        amplitude_to_DB = torchaudio.transforms.AmplitudeToDB("power", 80.0)  # 80.0 is the value of top dB
        mel_spec = MelSpec(input)
        log_mel = amplitude_to_DB(mel_spec)
        return log_mel

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
        
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1)
    for data in data_loader:
        continue
    a = jams.load("/import/c4dm-datasets/GuitarSet/annotation/00_BN1-129-Eb_comp.jams")
