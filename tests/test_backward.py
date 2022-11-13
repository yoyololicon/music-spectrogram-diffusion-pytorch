import torch
from models.diff_decoder import MIDI2SpecDiff
from models.ar_decoder import MIDI2SpecAR


def test_midi2spec_ar():
    model = MIDI2SpecAR(900, 128, 256, 64,
                        128, 6, 16, 6, 6, dim_feedforward=512)

    midi = torch.randint(0, 900, (10, 256))
    spec = torch.randn(10, 64, 128)
    y = model(midi, spec)
    loss = y.pow(2).mean()
    loss.backward()


def test_midi2spec_diff():
    model = MIDI2SpecDiff(900, 128, 256, 64,
                          128, 6, 16, 6, 6, with_context=False, dim_feedforward=512)

    midi = torch.randint(0, 900, (10, 256))
    spec = torch.randn(10, 64, 128)
    t = torch.rand(10)
    y = model(midi, spec, t)
    loss = y.pow(2).mean()
    loss.backward()

    model = MIDI2SpecDiff(900, 128, 256, 64,
                          128, 6, 16, 6, 6, with_context=True, dim_feedforward=512)
    context = torch.randn(10, 64, 128)
    y = model(midi, spec, t, context)
    loss = y.pow(2).mean()
    loss.backward()
