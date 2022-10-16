from tkinter import E
import pytest
import torch
from models.encoder import Encoder
from models.ar_decoder import Decoder, Transformer


def test_encoder():
    encoder = Encoder(6, 512, 6, 64, dim_feedforward=1024)

    x = torch.randn(10, 256, 512)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(256)
    x = encoder(x, mask)
    assert x.shape == (10, 256, 512)


def test_decoder():
    decoder = Decoder(6, 512, 6, 64, dim_feedforward=1024)

    x = torch.randn(10, 256, 512)
    y = torch.randn(10, 64, 512)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(64)
    y = decoder(y, x, tgt_mask=mask)
    assert y.shape == (10, 64, 512)


def test_transformer():
    transformer = Transformer(512, 6, 64, 6, 6, dim_feedforward=1024)

    x = torch.randn(10, 256, 512)
    y = torch.randn(10, 64, 512)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(64)
    y = transformer(x, y, tgt_mask=mask)
    assert y.shape == (10, 64, 512)
