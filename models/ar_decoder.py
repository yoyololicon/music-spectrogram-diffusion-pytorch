import torch
from torch import nn
import torch.nn.functional as F

from .encoder import geglu, Encoder, MultiheadAttention
from .utils import sinusoidal


class DecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, emb_dim, nhead, head_dim, dropout=0.1, **kwargs):
        super().__init__(emb_dim, 1, dropout=dropout, batch_first=True,  **kwargs)
        self.self_attn = MultiheadAttention(emb_dim, nhead, head_dim, dropout)
        self.multihead_attn = MultiheadAttention(
            emb_dim, nhead, head_dim, dropout)
        self.linear1 = nn.Linear(emb_dim, 2 * self.linear1.weight.shape[0])


class Decoder(nn.TransformerDecoder):
    def __init__(self, layers, emb_dim, nhead, head_dim, **kargs) -> None:
        decoder_layer = DecoderLayer(
            emb_dim, nhead, head_dim,
            activation=geglu, **kargs
        )
        decoder_norm = nn.LayerNorm(emb_dim)
        super().__init__(decoder_layer, layers, decoder_norm)


class Transformer(nn.Transformer):
    def __init__(self, emb_dim, nhead, head_dim,  num_encoder_layers, num_decoder_layers, **kwargs) -> None:
        encoder = Encoder(num_encoder_layers, emb_dim,
                          nhead, head_dim, **kwargs)
        decoder = Decoder(num_decoder_layers, emb_dim,
                          nhead, head_dim,  **kwargs)
        super().__init__(d_model=emb_dim, custom_decoder=decoder,
                         custom_encoder=encoder, batch_first=True,  **kwargs)


class MIDI2SpecAR(nn.Module):
    def __init__(self,
                 num_emb, output_dim,
                 max_input_length, max_output_length,
                 emb_dim, nhead, head_dim, num_encoder_layers, num_decoder_layers, **kwargs) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.register_buffer('in_pos_emb', sinusoidal(
            (max_input_length, emb_dim)))
        self.register_buffer('out_pos_emb', sinusoidal(
            (max_output_length, emb_dim)))
        self.transformer = Transformer(
            emb_dim, num_encoder_layers, num_decoder_layers, batch_first=True, **kwargs)
        self.linear_in = nn.Linear(output_dim, emb_dim)
        self.linear_out = nn.Linear(emb_dim, output_dim)

    def forward(self, midi_tokens, spec):
        # spec: (batch, seq_len, output_dim)
        # midi_tokens: (batch, seq_len)
        batch_size, seq_len = midi_tokens.shape
        midi = self.emb(midi_tokens) + self.in_pos_emb[:seq_len]
        spec = self.linear_in(spec) + self.out_pos_emb[:spec.shape[1]]
        spec_tri_mask = self.transformer.generate_square_subsequent_mask(
            spec.shape[1]).to(spec.device)
        x = self.transformer(midi, spec, tgt_mask=spec_tri_mask)
        x = self.linear_out(x)
        return x
