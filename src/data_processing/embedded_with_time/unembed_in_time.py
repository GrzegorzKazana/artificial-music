import mido
import numpy as np

from ..sparse_notes_quantized_time.config import DEFAULT_BPM
from ..sparse_notes_quantized_time.mid2np import note_off_to_zero_vel, secs_to_msecs, filter_meta, to_raw_numpy
from ..common.helpers import flow, debug
from ..embedding_sparse_notes.common import map_note_num_to_name, UNKNOWN_FRAME, reverse_midi_notes


def from_embedded_with_time(np_track, wv):
    """
    takes in [[embedded_note, velocity, d_time, duration], ...] and transforms to 
    [[note, velocity, delta_time, duration], ...], where
    embedded_note is vector from word vectors
    velocity is <0, 1>
    d_time is <0, ...> and 1 = quarter
    duration is <0, ...> and 1 = quarter
    Reverses 'to_embedded_with_time'
    """
    bpms = DEFAULT_BPM / 60 / 1000

    def decode(frame):
        note_code = frame[:-3]
        vel_code = frame[-3]
        d_time_code = frame[-2]
        duration_code = frame[-1]

        note_name = wv.similar_by_vector(note_code, topn=1)[0][0]
        # handle special tokens here

        note = reverse_midi_notes[note_name]
        vel = int(np.clip(vel_code * 127, 0, 127))
        d_time = d_time_code / bpms
        duration = duration_code / bpms

        return [note, vel, d_time, duration]

    return np.array([decode(f) for f in np_track])


def duration_to_note_offs(np_track):
    """
    [[note, vel, d_time, duration], ...] -> [[note, vel, time], ...]
    """
    acc_time = 0
    notes_abs_time = []

    # to absolute time
    for note, vel, d_time, duration in np_track:
        acc_time += d_time
        note_on_abs_time = acc_time
        note_off_abs_time = acc_time + duration

        notes_abs_time.append([note, vel, note_on_abs_time])
        notes_abs_time.append([note, 0, note_off_abs_time])

    notes_abs_time = sorted(notes_abs_time, key=lambda x: x[-1])

    notes_d_time = []
    now = 0
    # to delta time
    for note, vel, abs_time in notes_abs_time:
        delta = abs_time - now
        notes_d_time.append([note, vel, delta])
        now = abs_time

    return np.array(notes_d_time)


def np2mid(np_track, wv, embedding_dict):

    return flow(

    )(np_track)
