import math
import torch
from torch import nn, Tensor
from typing import Optional, Tuple
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


@torch.jit.script
def geglu(x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)


class MultiheadAttention(nn.Module):
    def __init__(self,  emb_dim, nhead, head_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.head_dim = head_dim
        self.num_heads = nhead

        self.q_proj = nn.Linear(emb_dim, head_dim * nhead, bias=False)
        self.k_proj = nn.Linear(emb_dim, head_dim * nhead, bias=False)
        self.v_proj = nn.Linear(emb_dim, head_dim * nhead, bias=False)
        self.emb_proj = nn.Linear(head_dim * nhead, emb_dim,  bias=False)

        xavier_uniform_(self.q_proj.weight.data)
        xavier_uniform_(self.k_proj.weight.data)
        xavier_uniform_(self.v_proj.weight.data)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        B, T, _ = query.size()
        Q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim)

        attn_score = Q @ K.transpose(-2, -1) / self.head_dim ** 0.5

        if attn_mask is not None:
            attn_score = attn_score + attn_mask.unsqueeze(1)

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = F.dropout(attn_prob, self.dropout, self.training)

        attn_vec = attn_prob.transpose(1, 2) @ V.transpose(1, 2)
        x = self.emb_proj(attn_vec.transpose(1, 2).reshape(B, T, -1))
        if not need_weights:
            return x, None
        if average_attn_weights:
            return x, attn_prob.mean(dim=2)
        return x, attn_prob


class EncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, emb_dim, nhead, head_dim, dropout=0.1, **kwargs):
        super().__init__(emb_dim, nhead, dropout, **kwargs)
        self.self_attn = MultiheadAttention(emb_dim, nhead, head_dim, dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


class Encoder(nn.TransformerEncoder):
    def __init__(self, layers, emb_dim, nhead, head_dim,  **kargs) -> None:
        encoder_layer = EncoderLayer(
            emb_dim, nhead, head_dim,
            activation=geglu, **kargs
        )
        encoder_norm = nn.LayerNorm(emb_dim)
        super().__init__(encoder_layer, layers, encoder_norm)
