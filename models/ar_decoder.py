import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

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
                         custom_encoder=encoder, batch_first=True, **kwargs)

    def autoregressive_infer(self, tgt, src=None, memory=None,
                             src_mask=None, tgt_mask=None, memory_mask=None,
                             src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if memory is None:
            memory = self.encoder(src, mask=src_mask,
                                  src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return out, memory


class MIDI2SpecAR(nn.Module):
    def __init__(self,
                 num_emb, output_dim,
                 max_input_length, max_output_length,
                 emb_dim, nhead, head_dim, num_encoder_layers, num_decoder_layers, **kwargs) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.register_buffer('in_pos_emb', sinusoidal(
            shape=(max_input_length, emb_dim)))
        self.register_buffer('out_pos_emb', sinusoidal(
            shape=(max_output_length, emb_dim)))
        self.transformer = Transformer(
            emb_dim, nhead, head_dim,  num_encoder_layers, num_decoder_layers, **kwargs)
        self.linear_in = nn.Linear(output_dim, emb_dim)
        self.linear_out = nn.Linear(emb_dim, output_dim)

    def forward(self, midi_tokens, spec):
        # spec: (batch, seq_len, output_dim)
        # midi_tokens: (batch, seq_len)
        batch_size, seq_len = midi_tokens.shape
        midi = self.emb(midi_tokens) + self.in_pos_emb[:seq_len]
        spec = self.linear_in(spec) + self.out_pos_emb[:spec.shape[1]]
        spec_tri_mask = self.transformer.generate_square_subsequent_mask(
            spec.shape[1], device=spec.device)
        x = self.transformer(midi, spec, tgt_mask=spec_tri_mask)
        x = self.linear_out(x)
        return x

    def infer(self, midi_tokens, max_len=512, dither_amount=0., verbose=True):
        batch_size, seq_len = midi_tokens.shape
        midi = self.emb(midi_tokens) + self.in_pos_emb[:seq_len]
        spec = midi.new_zeros(
            (batch_size, 1, self.linear_out.weight.shape[0]))
        memory = None

        for i in tqdm(range(max_len), disable=not verbose):
            spec_emb = self.linear_in(spec) + self.out_pos_emb[:i+1]
            next_spec, memory = self.transformer.autoregressive_infer(
                spec_emb, src=midi, memory=memory)
            next_spec = self.linear_out(next_spec[:, -1:])
            if dither_amount > 0:
                next_spec = next_spec + dither_amount * \
                    torch.randn_like(next_spec)
            next_spec.clamp_(-1, 1)
            spec = torch.cat([spec, next_spec], dim=1)

        return spec[:, 1:]
