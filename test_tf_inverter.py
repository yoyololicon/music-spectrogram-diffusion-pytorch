import math
import tensorflow as tf
import soundfile as sf
import torch
import librosa
from torchaudio.transforms import MelSpectrogram
import tensorflow_hub as hub

module = hub.KerasLayer(
    'https://tfhub.dev/google/soundstream/mel/decoder/music/1')

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 640
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8

mel = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MEL_CHANNELS,
    f_min=MEL_FMIN,
    f_max=MEL_FMAX,
    power=1.0,
)


def calculate_spectrogram(samples):
    mels = mel(samples)
    return mels.clamp(CLIP_VALUE_MIN, CLIP_VALUE_MAX).log()

y, sr = librosa.load(librosa.example('brahms'), sr=SAMPLE_RATE)
sf.write('brahms.wav', y, sr)
spectrogram = calculate_spectrogram(torch.tensor(y).unsqueeze(0))
spectrogram = tf.convert_to_tensor(spectrogram.transpose(1, 2).numpy())
# Reconstruct the audio from a mel-spectrogram using a SoundStream decoder.
reconstructed_samples = module(spectrogram)
sf.write('reconstructed.wav', reconstructed_samples.numpy().flatten(), sr)
