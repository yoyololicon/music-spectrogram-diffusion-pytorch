import dataclasses
import math
import note_seq
from .event_codec import EventRange, Codec


# defaults for vocabulary config
DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 127


@dataclasses.dataclass
class VocabularyConfig:
  """Vocabulary configuration parameters."""
  steps_per_second: int = DEFAULT_STEPS_PER_SECOND
  max_shift_seconds: int = DEFAULT_MAX_SHIFT_SECONDS
  num_velocity_bins: int = DEFAULT_NUM_VELOCITY_BINS


def build_codec(steps_per_second, max_shift_seconds, num_velocity_bins=1):
  """Build event codec."""
  event_ranges = [
      EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                             note_seq.MAX_MIDI_PITCH),
      # velocity bin 0 is used for note-off
      EventRange('velocity', 0, num_velocity_bins),
      # used to indicate that a pitch is present at the beginning of a segment
      # (only has an "off" event as when using ties all pitch events until the
      # "tie" event belong to the tie section)
      EventRange('tie', 0, 0),
      EventRange('program', note_seq.MIN_MIDI_PROGRAM,
                 note_seq.MAX_MIDI_PROGRAM),
      EventRange('drum', note_seq.MIN_MIDI_PITCH,
                note_seq.MAX_MIDI_PITCH),
  ]

  return Codec(
      max_shift_steps=(steps_per_second * max_shift_seconds),
      steps_per_second=steps_per_second,
      event_ranges=event_ranges)


def num_velocity_bins_from_codec(codec: Codec):
  """Get number of velocity bins from event codec."""
  lo, hi = codec.event_type_range('velocity')
  return hi - lo


def velocity_to_bin(velocity, num_velocity_bins):
  if velocity == 0:
    return 0
  else:
    return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


def bin_to_velocity(velocity_bin, num_velocity_bins):
  if velocity_bin == 0:
    return 0
  else:
    return int(note_seq.MAX_MIDI_VELOCITY * velocity_bin / num_velocity_bins)