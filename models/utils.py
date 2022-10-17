import torch
import math


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
	div_term = min_scale * torch.exp(torch.arange(0, features // 2) * scale_factor)
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
