from scipy import sparse
from collections import OrderedDict

from .common import remove_subsequent_notes, concat_tracks, hash_frame, map_note_num_to_name, map_hashed_frame_to_names


def count_note_occurences(npz_track):
    """
    creates a sorted dict of { [note_hash: string]: number }
    """
    counter = {}
    for time_step in npz_track:
        key = hash_frame(time_step)
        if key in counter:
            counter[key] += 1
        else:
            counter[key] = 1

    return OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))


def transform_counter_num_to_notes(counter):
    """
    transforms keys in counter to note names
    """
    counter = {map_hashed_frame_to_names(k): v for k, v in counter.items()}
    return OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))


def create_counters(npz_tracks):
    tracks = [t.tocsr() for t in npz_tracks]
    track_blob = concat_tracks(tracks)
    track_blob = remove_subsequent_notes(track_blob)
    counter = count_note_occurences(track_blob)

    return counter, transform_counter_num_to_notes(counter)
