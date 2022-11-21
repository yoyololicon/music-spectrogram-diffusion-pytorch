import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
import IPython.display

# init
pm = pretty_midi.PrettyMIDI(initial_tempo=80)  
# should use ["annotations"] if "namespace"== "tempo" then ["data"][0]["value"]

# midi instrument 
# index https://soundprogramming.net/file-formats/general-midi-instrument-list/
# guitarset mentioned acoustic guitar and steel string so should be 26th instrument in the link with index 25
inst = pretty_midi.Instrument(program=25, is_drum=False, name='acoustic guitar (steel)')
pm.instruments.append(inst)

inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
