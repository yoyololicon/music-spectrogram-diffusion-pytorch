from torch import nn
from torch.nn import functional as F

from .utils import get_timing_signal_1d
from .ar_decoder import DecoderLayer


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
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            tmp = self.norm3(x)
            tmp = self.film2(tmp, cond)
            x = x + self._ff_block(tmp)
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.film1(x, cond)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.film2(x, cond)
            x = self.norm3(x + self._ff_block(x))

        return x 
        
