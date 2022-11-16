import torch
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


def _audio_to_frames_pytorch(
    samples: torch.Tensor,
    hop_size: int,
    frame_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Convert audio samples to non-overlapping frames and frame times."""
  frame_size = hop_size
  samples = F.pad(samples,
                  [0, frame_size - len(samples) % frame_size],
                  mode='constant')

  # Split audio into frames.
  frames = samples.unfold(0, frame_size, frame_size) 

  num_frames = len(samples) // frame_size
  times = torch.arange(num_frames) / frame_rate
  return frames, times