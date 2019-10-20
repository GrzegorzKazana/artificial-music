import mido
import numpy as np

from ..sparse_notes_quantized_time.config import DEFAULT_BPM, NUM_NOTES, MSECS_PER_FRAME
from ..sparse_notes_quantized_time.mid2np import note_off_to_zero_vel, secs_to_msecs, filter_meta, msecs_to_frames, transform
from ..common.helpers import flow, debug
from ..embedding_sparse_notes.common import map_note_num_to_name, UNKNOWN_FRAME, TRACK_END, reverse_midi_notes


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

        note_token = wv.similar_by_vector(note_code, topn=1)[0][0]
        vel = int(np.clip(vel_code * 127, 0, 127))
        d_time = max(d_time_code / bpms, 0)
        duration = max(duration_code / bpms, 0)

        return [note_token, vel, d_time, duration]

    res = []
    for note_token, vel, d_time, duration in map(decode, np_track):
        if note_token == UNKNOWN_FRAME:
            continue
        elif note_token == TRACK_END:
            break
        else:
            res.append([reverse_midi_notes[note_token], vel, d_time, duration])
            print([reverse_midi_notes[note_token], vel, d_time, duration])

    return np.array(res)


def duration_to_note_offs(np_track):
    """
    [[note, vel, d_time, duration], ...] -> [[note, vel, time], ...]
    time in output notes is absolute
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

    return np.array(notes_abs_time)


def to_midi_messages(np_track):
    track_detlta_time = []
    now = 0
    for note, vel, abs_time in np_track:
        delta = int(abs_time - now)
        now = abs_time
        track_detlta_time.append([note, vel, delta])

    return [mido.Message('note_on', note=int(note), velocity=int(vel), time=int(abs_time))
            for note, vel, abs_time in track_detlta_time]


def messages_to_midi(messages):
    outfile = mido.MidiFile()
    track = mido.MidiTrack()
    outfile.tracks.append(track)
    for msg in messages:
        track.append(msg)

    return outfile


def to_sparse_matrix_rep(np_track, ms_per_frame, skip_velocities=True):
    return flow(
        lambda x: msecs_to_frames(x, ms_per_frame),
        lambda x: transform(x, skip_velocities),
    )(np_track)


def to_raw_numpy(np_track, wv):
    """
    [[note, vel, d_time, duration], ...] -> [[note, vel, abs_time], ...]
    """
    return flow(
        lambda x: from_embedded_with_time(x, wv),
        duration_to_note_offs,
    )(np_track)


def np2mid(np_track, wv):
    """
    [[note, vel, d_time, duration], ...] -> MidiFile
    """
    return flow(
        lambda x: to_raw_numpy(x, wv),
        to_midi_messages,
        messages_to_midi
    )(np_track)


def np2sparse(np_track, wv, ms_per_frame=MSECS_PER_FRAME):
    """
    [[note, vel, d_time, duration], ...] -> 
    [[one_hot_encoded_note], ...] (n_of_frames x one_hot_encoded_note)
    """
    return flow(
        lambda x: to_raw_numpy(x, wv),
        lambda x: to_sparse_matrix_rep(x, ms_per_frame)
    )(np_track)
