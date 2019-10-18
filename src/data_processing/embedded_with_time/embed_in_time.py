import mido
import numpy as np

from ..sparse_notes_quantized_time.config import DEFAULT_BPM
from ..sparse_notes_quantized_time.mid2np import note_off_to_zero_vel, secs_to_msecs, filter_meta, to_raw_numpy
from ..common.helpers import flow, debug
from ..embedding_sparse_notes.common import map_note_num_to_name, UNKNOWN_FRAME


def get_track_tempo(track):
    set_tempo_msgs = list(filter(lambda x: x.type == 'set_tempo', track))
    return mido.tempo2bpm(set_tempo_msgs[0].tempo) if len(set_tempo_msgs) > 0 else DEFAULT_BPM


def calc_notes_duration(track):
    """
    [[note, velocity, delta_time], ...] -> [[note, velocity, delta_time, duration], ...]
    """
    notes_w_durations = []
    for i, (note, vel, dtime) in enumerate(track):
        idxs_of_note_offs = np.argwhere(
            (track[i:, 1] == 0) & (track[i:, 0] == note)).reshape(-1)
        assert idxs_of_note_offs.size > 0, 'failed to find note_off'

        note_off_idx = idxs_of_note_offs[0]
        duration = track[i + 1: note_off_idx + 1, -1].sum()

        notes_w_durations.append([note, vel, dtime, duration])

    return np.array(notes_w_durations)


def to_embedded_with_time(raw_numpy, wv, embedding_dict, tempo):
    """
    takes in [[note, velocity, delta_time, duration], ...] and transforms to 
    [[embedded_note, velocity, d_time, duration], ...], where
    embedded_note is vector from word vectors
    velocity is <0, 1>
    d_time is <0, ...> and 1 = quarter
    duration is <0, ...> and 1 = quarter
    """
    def safe_dict_lookup(note_hashed):
        return note_hashed if note_hashed in embedding_dict else UNKNOWN_FRAME

    encoded = []
    bpms = tempo / 60 / 1000    # duration of quarter in ms

    for note, vel, d_time, duration in raw_numpy:
        note_code = wv[safe_dict_lookup(map_note_num_to_name(note))]
        vel_code = vel / 127
        d_time_code = d_time * bpms
        duration_code = duration * bpms

        code = [*note_code, vel_code, d_time_code, duration_code]
        encoded.append(code)

    return np.array(encoded)


def mid2np(track, wv, embedding_dict):
    """
    takes in midi track and word vectors embedding, returns encoded track in numpy format
    """
    track_tempo = get_track_tempo(track)

    return flow(
        note_off_to_zero_vel,
        secs_to_msecs,
        filter_meta,
        to_raw_numpy,
        calc_notes_duration,
        lambda x: to_embedded_with_time(x, wv, embedding_dict, track_tempo)
    )(track)
