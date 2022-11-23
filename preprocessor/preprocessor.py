import dataclasses
import json
import os
import time
import note_seq
import numpy as np
import torch
from typing import Any, Callable, MutableMapping, Optional, Sequence, Tuple, TypeVar
import torch.nn.functional as F

from . import vocabularies
from .event_codec import Codec, Event


folder = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(folder, "event_mapping.json"), "r") as f:
    EVENT_MAPPING = json.load(f)
    for type_ in EVENT_MAPPING.keys():
        EVENT_MAPPING[type_] = {int(k): v for k, v in EVENT_MAPPING[type_].items()}

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


def note_sequence_to_onsets_and_offsets_and_programs(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    # Sort by program and pitch and put offsets before onsets as a tiebreaker for
    # subsequent stable sort.
    notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
    times = [note.end_time for note in notes if not note.is_drum] + [
        note.start_time for note in notes
    ]
    values = [
        NoteEventData(pitch=note.pitch, velocity=0, program=note.program, is_drum=False)
        for note in notes
        if not note.is_drum
    ] + [
        NoteEventData(
            pitch=note.pitch,
            velocity=note.velocity,
            program=note.program,
            is_drum=note.is_drum,
        )
        for note in notes
    ]
    return times, values


def note_event_data_to_events(
    state: Optional[Any],
    value: NoteEventData,
    codec: Codec = None,
) -> Sequence[Event]:
    """Convert note event data to a sequence of events."""
    #num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
    velocity_bin = vocabularies.velocity_to_bin(value.velocity, 1)
    if value.is_drum:
        # drum events use a separate vocabulary
        # return [Event("velocity", velocity_bin), Event("drum", value.pitch)]
        return np.array([
                EVENT_MAPPING["velocity"][velocity_bin], 
                EVENT_MAPPING["drum"][value.pitch]
        ])
    else:
        # program + velocity + pitch
        if state is not None:
            state.active_pitches[(value.pitch, value.program)] = velocity_bin
        return [
            EVENT_MAPPING["program"][value.program],
            EVENT_MAPPING["velocity"][velocity_bin],
            EVENT_MAPPING["pitch"][value.pitch]
        ]


def note_encoding_state_to_events(state: NoteEncodingState, offset_value=EVENT_MAPPING['velocity'][0]) -> Sequence[Event]:
    """Output program and pitch events for active notes plus a final tie event."""
    events = []
    for pitch, program in sorted(state.active_pitches.keys(), key=lambda k: k[::-1]):
        if state.active_pitches[(pitch, program)] != offset_value:
            # events += [Event("program", program), Event("pitch", pitch)]
            events += [program, pitch]
    events.append(EVENT_MAPPING["tie"][0])
    return events


def read_midi(filename):
    with open(filename, "rb") as f:
        content = f.read()
        ns = note_seq.midi_to_note_sequence(content)
    return ns 


def tokenize(ns, frame_rate):
    notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
    times = [note.end_time*frame_rate for note in notes if not note.is_drum] + [
        note.start_time*frame_rate for note in notes
    ]
    def get_values(note):
        if note.is_drum:
            return [EVENT_MAPPING["velocity"][int(note.velocity > 0)], EVENT_MAPPING["pitch"][note.pitch]]
        return [EVENT_MAPPING["program"][note.program], EVENT_MAPPING["velocity"][int(note.velocity > 0)], EVENT_MAPPING["pitch"][note.pitch]]
    
    values = [
        [EVENT_MAPPING["program"][note.program], EVENT_MAPPING["velocity"][0], EVENT_MAPPING["pitch"][note.pitch]]
        for note in notes
        if not note.is_drum
    ] + [get_values(note) for note in notes]
    return np.round(times).astype(int), values


def quantize_time(times, frame_length):
    steps = np.round(np.array(times) / frame_length)
    return steps.astype(int)


def update_state(ds: NoteEncodingState, events):
    non_drum_idx = events[:, 0] != -1
    for event in events[non_drum_idx]:
        ds.active_pitches[(event[0], event[-1])] = event[1]

def preprocess(ns, resolution=100, segment_length=5.12, output_size=2048):
    segment_length = np.ceil(segment_length * resolution).astype(int)
    steps, values = tokenize(ns, resolution)
    stamps = np.unique(steps)
    num_segments = np.ceil(stamps[-1]/segment_length).astype(int)
    events = {}
    state_events = {0: [EVENT_MAPPING["tie"][0]]}
    ds = NoteEncodingState()
    segments, shifts = np.divmod(stamps, segment_length)
    change_points = np.zeros_like(segments)
    change_points[:-1] = segments[1:] != segments[:-1]
    for i, stamp in enumerate(stamps):
        segment_num, shift_num = segments[i], shifts[i]
        event_idx = np.nonzero(steps == stamp)[0]
        event_values = [values[i] for i in event_idx]
        event = events.get(segment_num, [])
        event = event + [shift_num] + [v for e in event_values for v in e]
        events[segment_num] = event
        for event in event_values:
            if len(event) == 3:
                ds.active_pitches[(event[0], event[-1])] = event[1]
        if (change_points[i]):
            state_event = note_encoding_state_to_events(ds)
            state_events[segment_num + 1] = state_event
    tokens = torch.zeros(num_segments, output_size)
    for k, v in events.items():
        all_events = torch.Tensor(state_events[k]+v)
        tokens[k][:len(all_events)] = all_events
    return tokens


def preprocess_torch(ns, resolution=100, segment_length=5.12, output_size=2048):
    segment_length = np.ceil(segment_length * resolution).astype(int)
    steps, values = tokenize(ns, resolution)
    values = torch.Tensor(values).int()
    steps = torch.Tensor(steps).int()
    stamps = torch.unique(steps)
    num_segments = np.ceil(stamps[-1].item()/segment_length).astype(int)
    events = torch.zeros(num_segments, output_size)
    event_count = torch.zeros(num_segments).int()
    state_events = torch.zeros(num_segments+1, output_size)
    state_events[0, 0] = EVENT_MAPPING["tie"][0]
    ds = NoteEncodingState()
    start_time = time.time()
    for stamp in stamps:
        segment_num = int(stamp // segment_length)
        event_idx = np.nonzero(steps == stamp)[0]
        event_values = values[event_idx]
        event_start_idx = event_count[segment_num]
        event_values = values[event_idx]
        valid_events = event_values != -1
        event_end_idx = int(event_start_idx + valid_events.sum() + 1)
        events[segment_num][event_start_idx] = stamp % segment_length
        events[segment_num][event_start_idx+1:event_end_idx] = event_values[valid_events]
        event_count[segment_num] = event_end_idx
        for event in event_values[valid_events[:, 0]]:
            ds.active_pitches[(event[0].item(), event[-1].item())] = event[1].item()
        state_event = note_encoding_state_to_events(ds)
        state_events[segment_num + 1] = F.pad(torch.Tensor(state_event), (0, output_size-len(state_event))) 
    print(time.time() - start_time)
    for i in range(num_segments):
        token = state_events[i]
        start_idx = (token != 0).sum()
        token[start_idx:start_idx+event_count[i]] = events[i][:event_count[i]] 
    return state_events[:-1] 
