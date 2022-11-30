import soundfile as sf
from pathlib import Path
import numpy as np
import note_seq
from tqdm import tqdm
import yaml
from random import sample, shuffle, random, randint
from typing import Tuple, Union, Optional, List, Dict, Any
from .common import Base


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
        stems_list = []
        print("Loading Slakh2100...")
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

            total_programs = list(program_stems_dict.keys())
            for _ in range(10):
                num_included_programs = randint(4, len(total_programs))
                included_programs = sample(
                    total_programs, num_included_programs)

                included_stems = []
                for program_num in included_programs:
                    included_stems += program_stems_dict[program_num]
                stems_list.append(included_stems)

                include_drums = 128 in included_programs
                if include_drums:
                    included_programs.remove(128)
                filtered_ns = _copy_notesequence(
                    ns, included_programs, include_drums)

                data_list.append((None, filtered_ns, sr, frames))

        super().__init__(data_list, **kwargs)

        self.stems_list = stems_list

    def _get_waveforms(self, index: int, chunk_index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        *_, sr, length_in_time = self.data_list[index]
        offset = int(chunk_index * length_in_time * sr)
        frames = int(length_in_time * sr)

        stems = self.stems_list[index]
        if not self.with_context:
            stems_chunk = []
            for stem in stems:
                data, _ = sf.read(
                    stem, start=offset, frames=frames, dtype='float32', always_2d=True)
                stems_chunk.append(data)
            data = sum(stems_chunk)
            data = data.mean(axis=1)
            return data

        ctx_offset = offset - frames
        if ctx_offset >= 0:
            stems_ctx = []
            for stem in stems:
                data, _ = sf.read(
                    stem, start=ctx_offset, frames=frames * 2, dtype='float32', always_2d=True)
                stems_ctx.append(data)
            ctx = sum(stems_ctx)
            data = ctx[frames:]
            ctx = ctx[:frames]
        else:
            stems_chunk = []
            for stem in stems:
                data, _ = sf.read(
                    stem, start=offset, frames=frames, dtype='float32', always_2d=True)
                stems_chunk.append(data)
            data = sum(stems_chunk)
            ctx = np.zeros_like(data)
        data = data.mean(axis=1)
        ctx = ctx.mean(axis=1)
        return data, ctx
