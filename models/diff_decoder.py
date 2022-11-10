import torch
from torch import nn
from torch.nn import functional as F

from .utils import get_timing_signal_1d
from .ar_decoder import DecoderLayer
from .encoder import geglu, Encoder


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps, emb_dim):
        super().__init__()
        self.max_steps = max_steps
        self.projection1 = nn.Linear(emb_dim, emb_dim * 4)
        self.projection2 = nn.Linear(emb_dim * 4, emb_dim * 4)

    def forward(self, t):
        emb = get_timing_signal_1d(
            t * self.max_steps, self.embedding.shape[1], max_timescale=self.max_steps)
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
        gamma += 1
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

    def forward(self, src, tgt, ctx, cond, src_mask=None, tgt_mask=None, ctx_mask=None,
                memory_mask=None, src_key_padding_mask=None, ctx_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        memory = self.encoder(src, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        if hasattr(self, 'context_encoder'):
            ctx = self.context_encoder(
                ctx, mask=ctx_mask, src_key_padding_mask=ctx_key_padding_mask)
            memory = torch.cat([memory, ctx], dim=1)

        output = self.decoder(tgt, memory, cond, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output
