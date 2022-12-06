import soundfile as sf
from pathlib import Path
import numpy as np
import note_seq
from tqdm import tqdm
import yaml
from random import sample, shuffle, random, randint
from typing import Tuple, Union, Optional, List, Dict, Any
from itertools import chain

from .common import Base


PIANO_PROGRAMS = (0, 1, 2, 3, 4, 5, 6, 7)
GUITAR_PROGRAMS = (24, 25, 26, 27, 28, 29, 30, 31)
BASS_PROGRAMS = (32, 33, 34, 35, 36, 37, 38, 39)

CERBERUS_COMBINATIONS = [
    {'piano'},
    {'guitar'},
    {'bass'},
    {'drums'},
    {'piano', 'drums'},
    {'guitar', 'drums'},
    {'bass', 'drums'},
    {'piano', 'guitar'},
    {'piano', 'bass'},
    {'guitar', 'bass'},
    {'piano', 'guitar', 'drums'},
    {'piano', 'bass', 'drums'},
    {'guitar', 'bass', 'drums'},
    {'piano', 'guitar', 'bass'},
    {'piano', 'guitar', 'bass', 'drums'},
]


def _copy_notesequence(midi: note_seq.NoteSequence, include_programs: List, include_drums: bool) -> note_seq.NoteSequence:
    new_midi = note_seq.NoteSequence()
    new_midi.ticks_per_quarter = midi.ticks_per_quarter
    new_midi.source_info.CopyFrom(midi.source_info)

    for ts in midi.time_signatures:
        time_signature = new_midi.time_signatures.add()
        time_signature.time = ts.time
        time_signature.numerator = ts.numerator
        time_signature.denominator = ts.denominator

    for ks in midi.key_signatures:
        key_signature = new_midi.key_signatures.add()
        key_signature.time = ks.time
        key_signature.key = ks.key
        key_signature.mode = ks.mode

    for tempo in midi.tempos:
        tempo_change = new_midi.tempos.add()
        tempo_change.time = tempo.time
        tempo_change.qpm = tempo.qpm

    for inst_info in midi.instrument_infos:
        instrument_info = new_midi.instrument_infos.add()
        instrument_info.instrument = inst_info.instrument
        instrument_info.name = inst_info.name

    for note in midi.notes:
        if note.is_drum and include_drums:
            new_note = new_midi.notes.add()
            new_note.CopyFrom(note)
        elif note.program in include_programs:
            new_note = new_midi.notes.add()
            new_note.CopyFrom(note)

    for pb in midi.pitch_bends:
        if pb.is_drum and include_drums:
            new_pb = new_midi.pitch_bends.add()
            new_pb.CopyFrom(pb)
        elif pb.program in include_programs:
            new_pb = new_midi.pitch_bends.add()
            new_pb.CopyFrom(pb)

    for cc in midi.control_changes:
        if cc.is_drum and include_drums:
            new_cc = new_midi.control_changes.add()
            new_cc.CopyFrom(cc)
        elif cc.program in include_programs:
            new_cc = new_midi.control_changes.add()
            new_cc.CopyFrom(cc)

    return new_midi


class Slakh2100(Base):

    def __init__(self,
                 path: str,
                 split: str = 'train',
                 **kwargs):

        path: Path = Path(path)
        if split == 'train':
            path = path / 'train'
        elif split == 'val':
            path = path / 'validation'
        elif split == 'test':
            path = path / 'test'
        else:
            raise ValueError(f'Invalid split: {split}')

        data_list = []
        # stems_list = []
        print("Loading Slakh2100 and Cerberus4 datasets...")
        for track_path in tqdm(list(path.iterdir())):
            if not track_path.is_dir():
                continue
            midi_file = track_path / 'all_src.mid'
            ns = note_seq.midi_file_to_note_sequence(midi_file)
            ns = note_seq.apply_sustain_control_changes(ns)
            mix_flac_file = track_path / 'mix.flac'
            info = sf.info(mix_flac_file)
            sr = info.samplerate
            frames = info.frames

            valid_stems = [x.stem for x in (
                track_path / 'stems').iterdir() if x.stem != 'mix']

            with open(track_path / 'metadata.yaml') as f:
                meta = yaml.safe_load(f)

            program_stems_dict = {}
            for stem_id, stem_v in meta['stems'].items():
                if stem_id not in valid_stems:
                    continue
                program_num = stem_v['program_num']
                if stem_v['is_drum']:
                    program_num = 128
                tmp = program_stems_dict.get(program_num, [])
                program_stems_dict[program_num] = tmp + \
                    [track_path / 'stems' / (stem_id + '.flac')]

            pianos_items = list(
                (k, v) for k, v in program_stems_dict.items() if k in PIANO_PROGRAMS)
            guitars_items = list(
                (k, v) for k, v in program_stems_dict.items() if k in GUITAR_PROGRAMS)
            bass_items = list(
                (k, v) for k, v in program_stems_dict.items() if k in BASS_PROGRAMS)
            drums_stems = program_stems_dict.get(128, [])

            cerberus_dict = {
                'piano': ([x[0] for x in pianos_items], list(y for x in pianos_items for y in x[1])),
                'guitar': ([x[0] for x in guitars_items], list(y for x in guitars_items for y in x[1])),
                'bass': ([x[0] for x in bass_items], list(y for x in bass_items for y in x[1])),
            }

            if split == 'test':
                data_list.append((mix_flac_file, ns, sr, frames))

                # Cerberus test set
                included_programs = cerberus_dict['piano'][0] + \
                    cerberus_dict['guitar'][0] + cerberus_dict['bass'][0]
                included_stems = cerberus_dict['piano'][1] + \
                    cerberus_dict['guitar'][1] + \
                    cerberus_dict['bass'][1] + drums_stems

                copyied_ns = _copy_notesequence(ns, included_programs, True)
                data_list.append((included_stems, copyied_ns, sr, frames))
                continue

            total_programs = list(program_stems_dict.keys())
            for _ in range(10):
                num_included_programs = randint(4, len(total_programs))
                included_programs = sample(
                    total_programs, num_included_programs)

                included_stems = []
                for program_num in included_programs:
                    included_stems += program_stems_dict[program_num]

                include_drums = 128 in included_programs
                if include_drums:
                    included_programs.remove(128)
                filtered_ns = _copy_notesequence(
                    ns, included_programs, include_drums)

                data_list.append((included_stems, filtered_ns, sr, frames))

            # Cerberus train/val set
            for instruments in CERBERUS_COMBINATIONS:
                included_programs = []
                included_stems = []
                include_drums = False
                for instrument in instruments:
                    if instrument == 'drums':
                        included_stems += drums_stems
                        include_drums = True
                    else:
                        included_programs += cerberus_dict[instrument][0]
                        included_stems += cerberus_dict[instrument][1]
                copyied_ns = _copy_notesequence(
                    ns, included_programs, include_drums)
                data_list.append((included_stems, copyied_ns, sr, frames))

        super().__init__(data_list, **kwargs)

    def _get_waveforms(self, index: int, chunk_index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        stems, _, sr, length_in_time = self.data_list[index]
        if type(stems) != list:
            return super()._get_waveforms(index, chunk_index)

        offset = int(chunk_index * length_in_time * sr)
        frames = int(length_in_time * sr)

        if not self.with_context:
            stems_chunk = []
            for stem in stems:
                try:
                    data, _ = sf.read(
                        stem, start=offset, frames=frames, dtype='float32', always_2d=True)
                    stems_chunk.append(data)
                except RuntimeError as e:
                    print(f'RuntimeError: {e}, {stem}')
            data = sum(stems_chunk)
            data = data.mean(axis=1)
            return data

        ctx_offset = offset - frames
        if ctx_offset >= 0:
            stems_ctx = []
            for stem in stems:
                try:
                    data, _ = sf.read(
                        stem, start=ctx_offset, frames=frames * 2, dtype='float32', always_2d=True)
                    stems_ctx.append(data)
                except RuntimeError as e:
                    print(f'RuntimeError: {e}, {stem}')
            ctx = sum(stems_ctx)
            data = ctx[frames:]
            ctx = ctx[:frames]
        else:
            stems_chunk = []
            for stem in stems:
                try:
                    data, _ = sf.read(
                        stem, start=offset, frames=frames, dtype='float32', always_2d=True)
                    stems_chunk.append(data)
                except RuntimeError as e:
                    print(f'RuntimeError: {e}, {stem}')
            data = sum(stems_chunk)
            ctx = np.zeros_like(data)
        data = data.mean(axis=1)
        ctx = ctx.mean(axis=1)
        return data, ctx
