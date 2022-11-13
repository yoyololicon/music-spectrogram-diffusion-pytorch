import pytest
import torch
from models.encoder import Encoder
from models.ar_decoder import Decoder, Transformer, MIDI2SpecAR
from models.diff_decoder import DiffDecoder, DiffTransformer, MIDI2SpecDiff


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


def test_diff_decoder():
    decoder = DiffDecoder(6, 512, 6, 64, 2048,
                          dim_feedforward=1024, norm_first=True)

    x = torch.randn(10, 256, 512)
    y = torch.randn(10, 64, 512)
    cond = torch.randn(10, 1, 2048)
    y = decoder(y, x, cond)
    assert y.shape == (10, 64, 512)


def test_transformer():
    transformer = Transformer(512, 6, 64, 6, 6, dim_feedforward=1024)

    x = torch.randn(10, 256, 512)
    y = torch.randn(10, 64, 512)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(64)
    y = transformer(x, y, tgt_mask=mask)
    assert y.shape == (10, 64, 512)


def test_diff_transformer():
    transformer = DiffTransformer(
        512, 6, 64, 2048, 6, 6, dim_feedforward=1024, norm_first=True)
    dropout_mask = torch.rand(10) > 0.5

    x = torch.randn(10, 256, 512)
    y = torch.randn(10, 64, 512)
    cond = torch.randn(10, 1, 2048)
    y = transformer(x, y, cond)
    assert y.shape == (10, 64, 512)
    y = transformer(x, y, cond, dropout_mask=dropout_mask)
    assert y.shape == (10, 64, 512)
    y = transformer(x, y, cond, dropout_mask=torch.ones_like(dropout_mask))
    assert y.shape == (10, 64, 512)
    y = transformer(x, y, cond, dropout_mask=torch.zeros_like(dropout_mask))
    assert y.shape == (10, 64, 512)

    transformer = DiffTransformer(
        512, 6, 64, 2048, 6, 6, dim_feedforward=1024, norm_first=True, with_context=True)
    context = torch.randn(10, 128, 512)
    y = transformer(x, y, cond, context)
    assert y.shape == (10, 64, 512)
    y = transformer(x, y, cond, context,
                    dropout_mask=torch.ones_like(dropout_mask))
    assert y.shape == (10, 64, 512)
    y = transformer(x, y, cond, context,
                    dropout_mask=torch.zeros_like(dropout_mask))
    assert y.shape == (10, 64, 512)


def test_midi2spec_ar():
    model = MIDI2SpecAR(900, 128, 256, 64,
                        512, 6, 64, 6, 6, dim_feedforward=1024)

    midi = torch.randint(0, 900, (10, 256))
    spec = torch.randn(10, 64, 128)
    y = model(midi, spec)
    assert y.shape == (10, 64, 128)


def test_midi2spec_diff():
    model = MIDI2SpecDiff(900, 128, 256, 64,
                          512, 6, 64, 6, 6, with_context=False, dim_feedforward=1024)

    midi = torch.randint(0, 900, (10, 256))
    spec = torch.randn(10, 64, 128)
    t = torch.rand(10)
    y = model(midi, spec, t)
    assert y.shape == (10, 64, 128)

    model = MIDI2SpecDiff(900, 128, 256, 64,
                          512, 6, 64, 6, 6, with_context=True, dim_feedforward=1024)
    context = torch.randn(10, 64, 128)
    y = model(midi, spec, t, context)
    assert y.shape == (10, 64, 128)