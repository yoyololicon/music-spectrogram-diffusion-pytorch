import torch
from torchaudio.transforms import MelSpectrogram

class MelFeature(MelSpectrogram):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).clamp(1e-5, 1e8).log().transpose(-1, -2)