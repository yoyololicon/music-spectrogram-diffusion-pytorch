import dataclasses
import note_seq
import numpy as np
import torch
from typing import (Any, Callable, MutableMapping, Optional, 
                    Sequence, Tuple, TypeVar)
import torch.nn.functional as F

from . import vocabularies
from .event_codec import Codec, Event


ES = TypeVar('ES', bound=Any)
DS = TypeVar('DS', bound=Any)
T = TypeVar('T', bound=Any)


@dataclasses.dataclass
class NoteEventData:
  pitch: int
  velocity: Optional[int] = None
  program: Optional[int] = None
  is_drum: Optional[bool] = None
  instrument: Optional[int] = None

@dataclasses.dataclass
class NoteEncodingState:
  """Encoding state for note transcription, keeping track of active pitches."""
  # velocity bin for active pitches and programs
  active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(
      default_factory=dict)


def note_sequence_to_onsets_and_offsets_and_programs(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
  # Sort by program and pitch and put offsets before onsets as a tiebreaker for
  # subsequent stable sort.
  notes = sorted(ns.notes,
                 key=lambda note: (note.is_drum, note.program, note.pitch))
  times = ([note.end_time for note in notes if not note.is_drum] +
           [note.start_time for note in notes])
  values = ([NoteEventData(pitch=note.pitch, velocity=0,
                           program=note.program, is_drum=False)
             for note in notes if not note.is_drum] +
            [NoteEventData(pitch=note.pitch, velocity=note.velocity,
                           program=note.program, is_drum=note.is_drum)
             for note in notes])
  return times, values


def encode_and_index_events(
    state: DS,
    event_times: Sequence[float],
    event_values: Sequence[T],
    encode_event_fn: Callable[[ES, T, Codec],
                              Sequence[Event]],
    codec: Codec,
    frame_times: Sequence[float],
    encoding_state_to_events_fn: Optional[
        Callable[[ES], Sequence[Event]]] = None,
) -> Tuple[Sequence[int], Sequence[int], Sequence[int],
           Sequence[int], Sequence[int]]:

  indices = np.argsort(event_times, kind='stable')
  event_steps = [round(event_times[i] * codec.steps_per_second)
                 for i in indices]
  event_values = [event_values[i] for i in indices]

  events = []
  state_events = []
  event_start_indices = []
  state_event_indices = []

  cur_step = 0
  cur_event_idx = 0
  cur_state_event_idx = 0

  def fill_event_start_indices_to_cur_step():
    while(len(event_start_indices) < len(frame_times) and
          frame_times[len(event_start_indices)] <
          cur_step / codec.steps_per_second):
      event_start_indices.append(cur_event_idx)
      state_event_indices.append(cur_state_event_idx)

  for event_step, event_value in zip(event_steps, event_values):
    while event_step > cur_step:
      events.append(codec.encode_event(Event(type='shift', value=1)))
      cur_step += 1
      fill_event_start_indices_to_cur_step()
      cur_event_idx = len(events)
      cur_state_event_idx = len(state_events)
    if encoding_state_to_events_fn:
      # Dump state to state events *before* processing the next event, because
      # we want to capture the state prior to the occurrence of the event.
      for e in encoding_state_to_events_fn(state):
        state_events.append(codec.encode_event(e))
    for e in encode_event_fn(state, event_value, codec):
      events.append(codec.encode_event(e))

  # After the last event, continue filling out the event_start_indices array.
  # The inequality is not strict because if our current step lines up exactly
  # with (the start of) an audio frame, we need to add an additional shift event
  # to "cover" that frame.
  while cur_step / codec.steps_per_second <= frame_times[-1]:
    events.append(codec.encode_event(Event(type='shift', value=1)))
    cur_step += 1
    fill_event_start_indices_to_cur_step()
    cur_event_idx = len(events)

  # Now fill in event_end_indices. We need this extra array to make sure that
  # when we slice events, each slice ends exactly where the subsequent slice
  # begins.
  event_end_indices = event_start_indices[1:] + [len(events)]

  events = torch.Tensor(events).int()
  state_events = torch.Tensor(state_events).int()
  event_start_indices = torch.Tensor(event_start_indices).int()
  event_end_indices = torch.Tensor(event_end_indices).int()
  state_event_indices = torch.Tensor(state_event_indices).int()

  return (events, event_start_indices, event_end_indices,
          state_events, state_event_indices)


def note_event_data_to_events(
    state: Optional[Any],
    value: NoteEventData,
    codec: Codec,
) -> Sequence[Event]:
  """Convert note event data to a sequence of events."""
  num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
  velocity_bin = vocabularies.velocity_to_bin(
      value.velocity, num_velocity_bins)
  if value.is_drum:
    # drum events use a separate vocabulary
    return [Event('velocity', velocity_bin),
            Event('drum', value.pitch)]
  else:
    # program + velocity + pitch
    if state is not None:
      state.active_pitches[(value.pitch, value.program)] = velocity_bin
    return [Event('program', value.program),
            Event('velocity', velocity_bin),
            Event('pitch', value.pitch)]


def note_encoding_state_to_events(
    state: NoteEncodingState
) -> Sequence[Event]:
  """Output program and pitch events for active notes plus a final tie event."""
  events = []
  for pitch, program in sorted(
      state.active_pitches.keys(), key=lambda k: k[::-1]):
    if state.active_pitches[(pitch, program)]:
      events += [Event('program', program),
                 Event('pitch', pitch)]
  events.append(Event('tie', 0))
  return events


def split_tokens(features, segment_length=256):
    num_segments = np.ceil(len(features[0])/segment_length).astype(int)
    paddings = int(num_segments * segment_length - len(features[0]))
    results = []
    for feature in features:
        padded = F.pad(feature, (0, paddings), value=feature[-1])
        results.append(padded.view(num_segments, segment_length))
    return results


def extract_sequence_with_indices(events, start_idx, end_idx, 
                                  state_events,
                                  state_events_end_token,
                                  state_event_indices):
  """Extract target sequence corresponding to audio token segment."""
  segmented_events = events[start_idx:end_idx]

  if state_events_end_token is not None:
    # Extract the state events corresponding to the audio start token, and
    # prepend them to the targets array.
    state_event_start_idx = state_event_indices[0]
    state_event_end_idx = state_event_start_idx + 1
    while state_events[state_event_end_idx - 1] != state_events_end_token:
      state_event_end_idx += 1
    segmented_events = torch.concat((
        state_events[state_event_start_idx:state_event_end_idx],
        segmented_events
    ), dim=0)

  return segmented_events


def count_shift_and_pad(
    event_segment: torch.Tensor, 
    output_size: int,
    codec: Codec
) -> torch.Tensor:
    has_shift = False
    total_shift_steps = 0
    current_idx = 0
    padded_events = torch.zeros(output_size)
    for e in event_segment:
      if codec.is_shift_event_index(e):
        has_shift  = True
        total_shift_steps += 1
      else:
        if has_shift:
          padded_events[current_idx] = total_shift_steps 
          has_shift = False
          current_idx += 1
        padded_events[current_idx] = e
        current_idx += 1
    return padded_events


def read_midi(filename):
    with open(filename, 'rb') as f:
        content = f.read()
        ns = note_seq.midi_to_note_sequence(content)
    return ns 

def tokenize(filename, frame_rate, segment_length, output_size, step_rate=100):
    codec = vocabularies.build_codec(step_rate, segment_length/frame_rate)
    ns = read_midi(filename)
    ns = note_seq.apply_sustain_control_changes(ns)
    num_frames = np.ceil(ns.total_time * frame_rate)
    frame_times = torch.arange(num_frames) / frame_rate
    times, values = note_sequence_to_onsets_and_offsets_and_programs(ns)
    (events, event_start_indices, event_end_indices, 
    state_events, state_event_indices) = encode_and_index_events(
        NoteEncodingState(), 
        times, values, note_event_data_to_events,
        codec, frame_times, note_encoding_state_to_events
    )

    seg_start_idx, seg_end_idx, seg_state_idx = split_tokens([
        torch.Tensor(event_start_indices), 
        torch.Tensor(event_end_indices), 
        torch.Tensor(state_event_indices),
    ], segment_length=segment_length)

    tie_end_token = codec.encode_event(Event("tie", 0))

    segmented_events = torch.zeros(len(seg_start_idx), output_size)
    for i in range(len(seg_start_idx)):
        event = extract_sequence_with_indices(
            events, seg_start_idx[i, 0], seg_end_idx[i, -1], 
            torch.Tensor(state_events), tie_end_token, 
            seg_state_idx[i]
        )
        segmented_events[i] = count_shift_and_pad(event, output_size, codec)
    return segmented_events
