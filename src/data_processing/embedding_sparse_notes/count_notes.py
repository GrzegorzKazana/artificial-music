from scipy import sparse
from collections import OrderedDict

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
midi_notes = {i: f'{notes[i % 12]}{(i // 12) - 1}' for i in range(128)}


def map_note_num_to_name(note_num):
    """
    i.e. 72 -> 'C6'
    """
    return midi_notes[note_num]


def map_hashed_frame_to_names(frame_hash):
    """
    i.e. '72,75' -> 'C6,D#6'
    """
    if frame_hash == '':
        return ''
    note_nums = [int(n) for n in frame_hash.split(',')]
    note_names = map(map_note_num_to_name, note_nums)
    return ','.join(note_names)


def remove_subsequent_notes(npz_track):
    """
    takes in sparse matrix (frames x 128)
    and removes consecutive occurences of notes 
    """
    def notes_eq(noteA, noteB):
        return (noteA != noteB).nnz == 0

    acc = [npz_track[0]]
    for row_idx in range(1, npz_track.shape[0]):
        if not notes_eq(npz_track[row_idx], acc[-1]):
            acc.append(npz_track[row_idx])

    return sparse.vstack(acc)


def concat_tracks(npz_tracks):
    """
    joins tracks [(frames x 128), ...] -> (framesSum x 128)
    """
    return sparse.vstack(npz_tracks)


def hash_frame(npz_note):
    """
    maps one frame (1 x 128) to string of index of all non zero elements
    """
    return ','.join([str(n) for n in npz_note.nonzero()[1]])


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
    track_blob = concat_tracks(npz_tracks)
    track_blob = remove_subsequent_notes(track_blob)
    counter = count_note_occurences(track_blob)

    return counter, transform_counter_num_to_notes(counter)
