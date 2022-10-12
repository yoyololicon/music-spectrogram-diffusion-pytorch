import torch
from torch import nn
import torch.nn.functional as F


def geglu(x):
    x, gates = x.chunk(2, dim = -1)
    return x * F.gelu(gates)

class Encoder(nn.TransformerEncoder):
    def __init__(self, layers, nhead, d_model, dim_feedforward) -> None:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            activation=geglu
        )
        encoder_norm = nn.LayerNorm(d_model)
        super().__init__(encoder_layer, layers, encoder_norm)

