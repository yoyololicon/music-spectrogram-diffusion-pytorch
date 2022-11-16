import note_seq
from .event_codec import EventRange, Codec


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
