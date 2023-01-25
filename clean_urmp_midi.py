import mido
import os
import glob
import argparse

midi_mapping = {
    'tpt': 56,
    'tbn': 57,
    'tba': 58,
    'vn': 40,
    'va': 41,
    'vc': 42,
    'db': 43,
    'fl': 73,
    'cl': 71,
    'ob': 68,
    'sax': 65, 
    'bn': 70,
    'hn': 60,
}

def main(urmp_base, new_base):
    for midifile in glob.glob(os.path.join(urmp_base, '**/*.mid')):
        m = mido.MidiFile(midifile)
        title = os.path.basename(os.path.dirname(midifile))
        instruments = title.split('_')[2:]
        assert len(instruments) == len(m.tracks) - 1, f"{title} MIDI file does not have expected number of tracks" # expecting first track to be meta information
        
        for i in range(len(instruments)):
            assert m.tracks[i+1][0].type == "program_change"
            m.tracks[i+1][0].program = midi_mapping[instruments[i]]
        
        m.save(os.path.join(new_base, title + ".mid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('orig_urmp_dir')
    parser.add_argument('clean_midi_output_dir')
    args = parser.parse_args()
    main(args.orig_urmp_dir, args.clean_midi_output_dir)