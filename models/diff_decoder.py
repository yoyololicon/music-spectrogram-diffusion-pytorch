import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

from .utils import get_timing_signal_1d
from .ar_decoder import DecoderLayer
from .encoder import geglu, Encoder
from .utils import sinusoidal


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps, emb_dim, cond_dim):
        super().__init__()
        self.max_steps = max_steps
        self.emb_dim = emb_dim
        self.projection1 = nn.Linear(emb_dim, cond_dim)
        self.projection2 = nn.Linear(cond_dim, cond_dim)

    def forward(self, t):
        emb = get_timing_signal_1d(
            t * self.max_steps, self.emb_dim, max_timescale=self.max_steps)
        emb = self.projection1(emb)
        emb = F.silu(emb)
        emb = self.projection2(emb)
        emb = F.silu(emb)
        return emb


class FiLM(nn.Module):
    def __init__(self, cond_dim, emb_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, emb_dim * 2)

    def forward(self, x, emb):
        cond = self.linear(emb)
        gamma, beta = cond.chunk(2, dim=-1)
        gamma = 1 + gamma
        return gamma * x + beta


class DiffDecoderLayer(DecoderLayer):
    def __init__(self, emb_dim, nhead, head_dim, cond_dim, dropout=0.1, **kwargs):
        super().__init__(emb_dim, nhead, head_dim, dropout=dropout, **kwargs)
        self.film1 = FiLM(cond_dim, emb_dim)
        self.film2 = FiLM(cond_dim, emb_dim)

    def forward(self, tgt, memory, cond, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt

        if self.norm_first:
            tmp = self.norm1(x)
            tmp = self.film1(tmp, cond)
            x = x + self._sa_block(tmp, tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory,
                                    memory_mask, memory_key_padding_mask)
            tmp = self.norm3(x)
            tmp = self.film2(tmp, cond)
            x = x + self._ff_block(tmp)
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.film1(x, cond)
            x = self.norm2(x + self._mha_block(x, memory,
                           memory_mask, memory_key_padding_mask))
            x = self.film2(x, cond)
            x = self.norm3(x + self._ff_block(x))

        return x


class DiffDecoder(nn.TransformerDecoder):
    def __init__(self, layers, emb_dim, nhead, head_dim, cond_dim, **kargs) -> None:
        decoder_layer = DiffDecoderLayer(
            emb_dim, nhead, head_dim, cond_dim,
            activation=geglu, **kargs
        )
        decoder_norm = nn.LayerNorm(emb_dim)
        super().__init__(decoder_layer, layers, decoder_norm)

    def forward(self, tgt, memory, cond, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, cond,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DiffTransformer(nn.Module):
    def __init__(self, emb_dim, nhead, head_dim, cond_dim, num_encoder_layers, num_decoder_layers, with_context=False, **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(num_encoder_layers, emb_dim,
                               nhead, head_dim, **kwargs)
        self.decoder = DiffDecoder(num_decoder_layers, emb_dim,
                                   nhead, head_dim,  cond_dim, **kwargs)

        if with_context:
            self.context_encoder = Encoder(
                num_encoder_layers, emb_dim, nhead, head_dim, **kwargs)

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, cond, ctx=None, dropout_mask=None, src_mask=None, tgt_mask=None, ctx_mask=None,
                memory_mask=None, src_key_padding_mask=None, ctx_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        if dropout_mask is not None:
            memory = torch.zeros_like(src)
            ctx_memory = torch.zeros_like(ctx) if ctx is not None else None
            if not dropout_mask.all():
                memory[~dropout_mask] = self.encoder(src[~dropout_mask], mask=src_mask,
                                                     src_key_padding_mask=src_key_padding_mask)

                if ctx is not None:
                    ctx_memory[~dropout_mask] = self.context_encoder(
                        ctx[~dropout_mask], mask=ctx_mask, src_key_padding_mask=ctx_key_padding_mask)
        else:
            memory = self.encoder(src, mask=src_mask,
                                  src_key_padding_mask=src_key_padding_mask)
            ctx_memory = None
            if ctx is not None:
                ctx_memory = self.context_encoder(
                    ctx, mask=ctx_mask, src_key_padding_mask=ctx_key_padding_mask)

        if ctx_memory is not None:
            memory = torch.cat([memory, ctx_memory], dim=1)

        output = self.decoder(tgt, memory, cond, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class MIDI2SpecDiff(nn.Module):
    def __init__(self,
                 num_emb, output_dim,
                 max_input_length, max_output_length,
                 emb_dim, nhead, head_dim, num_encoder_layers, num_decoder_layers, with_context=False, **kwargs) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.register_buffer('in_pos_emb', sinusoidal(
            shape=(max_input_length, emb_dim), permute_bands=True, random_phase_offsets=True))
        self.register_buffer('out_pos_emb', sinusoidal(
            shape=(max_output_length, emb_dim), permute_bands=True, random_phase_offsets=True))
        self.transformer = DiffTransformer(
            emb_dim, nhead, head_dim,  emb_dim * 4, num_encoder_layers, num_decoder_layers, with_context=with_context, **kwargs)
        self.linear_in = nn.Linear(output_dim, emb_dim)
        self.linear_out = nn.Linear(emb_dim, output_dim)
        self.diffusion_emb = DiffusionEmbedding(2e4, emb_dim, emb_dim * 4)

    def forward(self, midi_tokens, spec, t, context=None, **kwargs):
        # spec: (batch, seq_len, output_dim)
        # midi_tokens: (batch, seq_len)
        batch_size, seq_len = midi_tokens.shape
        midi = self.emb(midi_tokens) + self.in_pos_emb[:seq_len]
        spec = self.linear_in(spec) + self.out_pos_emb[:spec.shape[1]]
        diff_cond = self.diffusion_emb(t).unsqueeze(1)
        if context is not None:
            context = self.linear_in(context) + \
                self.out_pos_emb[:context.shape[1]]
        x = self.transformer(midi, spec, diff_cond, context, **kwargs)
        x = self.linear_out(x)
        return x
