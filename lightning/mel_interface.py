import torch
from torchaudio.transforms import MelSpectrogram

class MelFeatureInterface(object):
    mel: MelSpectrogram

    def get_mel(self, x: torch.Tensor) -> torch.Tensor:
        return self.mel(x).clamp(1e-5, 1e8).log().transpose(-1, -2)