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
import torch
import torchaudio
import jams
import os
import tqdm
import pickle
import glob

SR = 44100


class MyDataset(torch.utils.data.Dataset): #padding等加在getterm #pad放在init #np_to_torch放在
    def __init__(self, path="/import/c4dm-datasets/MAPS_working/MAPS"):
        self.path = path
        self.data = []
        try:
            file_ = open('./MAPS_data.pk', 'wb')
            self.data = pickle.load(file_)
            file_.close()
        except:
            file_ = open('./MAPS_data.pk', 'wb')
            file_dirs = os.walk(path)
            for file_dir, _, __ in tqdm.tqdm(file_dirs):
                # if len(self.data) > 10:
                #     break
                file_names = glob.glob(file_dir+'/*.wav') 
                for file in file_names:
                    tmp, sr = torchaudio.load(file)
                    title = file[:-4]
                    duration = tmp.shape[1]
                    num_seq = int(duration * 100 + 511) // 512
                    self.data.extend([(title, i) for i in range(num_seq)])
                
            pickle.dump(self.data, file_)
            file_.close()
        
    def __len__(self): # 返回df的长度
        return len(self.data)
    
    def __getitem__(self, idx): # 获取第idx+1列的数据
        title, time = self.data[idx]
        audio, sr = torchaudio.load(f"{title}.wav", frame_offset=int(SR*5.12*time), num_frames=int(SR*5.12))
        if time==0:
            context = torch.zeros((1, int(5.12*sr)))
        else:
            context, sr = torchaudio.load(f"{self.path}/audio_mono-pickup_mix/{title}_mix.wav", frame_offset=int(SR*5.12*(time-1)), num_frames=int(SR*5.12))
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = transform(audio)
        context = transform(context)
        audio = audio.mean(0)
        if audio.shape[0] < 5.12 * 16000:
            audio = torch.nn.functional.pad(audio, (0, int(5.12 * 16000 - audio.shape[1])))
        midi_path = f"{title}.mid"
        return midi_path, context, audio


if __name__ == "__main__":
    data_set = MyDataset()
        
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1)
    for data in data_loader:
        print(data[0], data[1].shape, data[2].shape)
        breaks
