# Copyright 2022 The Music Spectrogram Diffusion Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encode and decode events."""

import dataclasses
from typing import Tuple, Optional, MutableMapping, MutableSet
import torch
import json
import os
import note_seq

@dataclasses.dataclass
class EventRange:
    type: str
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: str
    value: int


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
        default_factory=dict
    )


@dataclasses.dataclass
class NoteDecodingState:
  """Decoding state for note transcription."""
  current_time: float = 0.0
  # velocity to apply to subsequent pitch events (zero for note-off)
  current_velocity: int = 100 
  # program to apply to subsequent pitch events
  current_program: int = 0
  # onset time and velocity for active pitches and programs
  active_pitches: MutableMapping[Tuple[int, int],
                                 Tuple[float, int]] = dataclasses.field(
                                     default_factory=dict)
  # pitches (with programs) to continue from previous segment
  tied_pitches: MutableSet[Tuple[int, int]] = dataclasses.field(
      default_factory=set)
  # whether or not we are in the tie section at the beginning of a segment
  is_tie_section: bool = False
  # partially-decoded NoteSequence
  note_sequence: note_seq.NoteSequence = dataclasses.field(
      default_factory=lambda: note_seq.NoteSequence(ticks_per_quarter=220))


class Codec:
    """Encode and decode events.

    Useful for declaring what certain ranges of a vocabulary should be used for.
    This is intended to be used from Python before encoding or after decoding with
    GenericTokenVocabulary. This class is more lightweight and does not include
    things like EOS or UNK token handling.

    To ensure that 'shift' events are always the first block of the vocab and
    start at 0, that event type is required and specified separately.
    """

    def __init__(self):
        """Define Codec.

        Args:
          max_shift_steps: Maximum number of shift steps that can be encoded.
          steps_per_second: Shift steps will be interpreted as having a duration of
              1 / steps_per_second.
          event_ranges: Other supported event types and their ranges.
        """
        folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(folder, "event_mapping.json"), "r") as f:
            self.encoding_mapping = json.load(f)
            for type_ in self.encoding_mapping.keys():
                self.encoding_mapping[type_] = {int(k): v for k, v in self.encoding_mapping[type_].items()}
            self.decode_mapping = {v: (k, t) for t, m in self.encoding_mapping.items() for k, v in m.items()}

    def is_shift_event_index(self, index: int) -> bool:
        return self.decode_mapping[index][1] == 'shift' 

    def is_shift_event_index_torch(self, index: int) -> bool:
        return torch.logical_and(
            (self._shift_range.min_value <= index),
            (index <= self._shift_range.max_value),
        )


    def encode_note(self, note: NoteEventData, velocity: int=None) -> int:
        """Encode an event to an index."""
        velocity = int(note.velocity > 0) if velocity is None else velocity 
        if note.is_drum:
            return [
                    self.encoding_mapping["velocity"][velocity], 
                    self.encoding_mapping["drum"][note.pitch]
            ]
        else:
            return [
                self.encoding_mapping["program"][note.program],
                self.encoding_mapping["velocity"][velocity],
                self.encoding_mapping["pitch"][note.pitch]
            ]

    def encode_event(self, event: Event):
        return self.encoding_mapping[event.type][event.value]

    def decode_event_index(self, index: int) -> Event:
        """Decode an event index to an Event."""
        index = int(index.item())
        value, type = self.decode_mapping[index]
        return Event(type=type, value=value)