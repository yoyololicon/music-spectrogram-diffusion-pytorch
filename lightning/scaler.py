import torch
from torch import nn
import math


class Scaler(nn.Module):
    def __init__(self, init_min: float = math.inf, init_max: float = -math.inf):
        super().__init__()
        self.register_buffer("min", torch.tensor(init_min))
        self.register_buffer("max", torch.tensor(init_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min) / (self.max - self.min)


def adaptive_update_hook(module: Scaler, input):
    x = input[0]
    if module.training:
        module.min.fill_(torch.min(module.min, x.min()))
        module.max.fill_(torch.max(module.max, x.max()))


def get_scaler(adaptive: bool = True, **kwargs) -> Scaler:
    scaler = Scaler(**kwargs)
    if adaptive:
        scaler.register_forward_pre_hook(adaptive_update_hook)
    return scaler
