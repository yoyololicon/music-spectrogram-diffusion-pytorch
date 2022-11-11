from numpy import dtype
import torch
import math


def get_timing_signal_1d(position: torch.Tensor,
                         num_channels: int,
                         min_timescale: float = 1.0,
                         max_timescale: float = 2.0e4) -> torch.Tensor:
    """Returns the positional encoding (same as Tensor2Tensor).

    Args:
      position: An array of shape [batch_size].
      num_channels: The number of output channels.
      min_timescale: The smallest time unit (should probably be 0.0).
      max_timescale: The largest time unit.

    Returns:
      a Tensor of timing signals [1, length, num_channels]
    """
    assert position.ndim == 1
    assert num_channels % 2 == 0
    dtype, device = position.dtype, position.device
    num_timescales = num_channels / 2
    log_timescale_increment = (
        math.log(max_timescale / min_timescale) / (num_timescales - 1.0))
    inv_timescales = min_timescale * \
        torch.exp(torch.arange(num_timescales, device=device, dtype=dtype) * -log_timescale_increment)
    scaled_time = position[:, None] * inv_timescales[None, :]
    signal = torch.view_as_real(
        torch.exp(1j * scaled_time)).view(position.shape[0], num_channels)
    return signal


def sinusoidal(min_scale: float = 1.0,
               max_scale: float = 10000.0,
               shape: tuple = (512, 512),
               permute_bands: bool = False,
               random_phase_offsets: bool = False):
    """Creates 1D Sinusoidal Position Embedding Initializer.

    Args:
            min_scale: Minimum frequency-scale in sine grating.
            max_scale: Maximum frequency-scale in sine grating.
            dtype: The DType of the returned values.
            permute_bands: If True, sinusoid band order will be randomly permuted at
            initialization.
            random_phase_offsets: If True, each band's phase will have a random offset
            at initialization.

    Returns:
            The sinusoidal initialization function.
    """
    max_len, features = shape
    position = torch.arange(0, max_len).unsqueeze(1)
    scale_factor = -math.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * \
        torch.exp(torch.arange(0, features // 2) * scale_factor)
    rads = position * div_term
    if random_phase_offsets:
        sin_offsets = torch.rand(features // 2) * 2 * math.pi
        cos_offsets = torch.rand(features // 2) * 2 * math.pi
    else:
        sin_offsets = 0.
        cos_offsets = 0.
    pe = torch.empty(max_len, features, dtype=rads.dtype)
    pe[:, :features // 2] = torch.sin(rads + sin_offsets)
    pe[:, features // 2:] = torch.cos(rads + cos_offsets)
    if permute_bands:
        pe = pe[:, torch.randperm(features)]
    return pe
