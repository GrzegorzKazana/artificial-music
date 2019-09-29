from scipy import sparse
import numpy as np

# special tokens
UNKNOWN_FRAME = '<UNKNOWN>'
TRACK_END = '<TRACK_END>'

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
midi_notes = {i: f'{notes[i % 12]}{(i // 12) - 1}' for i in range(128)}
reverse_midi_notes = {v: k for k, v in midi_notes.items()}


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


def remove_subsequent_dict_values(track_dicted):
    """
    takes in 1d np.array
    and removes consecutive occurences of values
    """
    def notes_eq(noteA, noteB):
        return noteA == noteB

    acc = [track_dicted[0]]
    for row_idx in range(1, track_dicted.shape[0]):
        if not notes_eq(track_dicted[row_idx], acc[-1]):
            acc.append(track_dicted[row_idx])

    return np.array(acc)


def concat_tracks(npz_tracks):
    """
    joins tracks [(frames x 128), ...] -> (framesSum x 128)
    """
    return sparse.vstack(npz_tracks)


def map_note_num_to_name(note_num):
    """
    i.e. 72 -> 'C5'
    """
    return midi_notes[note_num]


def map_hashed_frame_to_names(frame_hash):
    """
    i.e. '72,75' -> 'C5,D#5'
    """
    if frame_hash == '':
        return ''
    note_nums = [int(n) for n in frame_hash.split(',')]
    note_names = map(map_note_num_to_name, note_nums)
    return ','.join(note_names)


def hash_frame(npz_note):
    """
    maps one frame (1 x 128) to string of index of all non zero elements
    """
    return ','.join([str(n) for n in npz_note.nonzero()[1]])


def unhash_named_frame(hashed_frame):
    """
    i.e. 'C6,D#6' -> np.array (1 x 128)
    """
    res = np.zeros((1, 128))

    if hashed_frame == '':
        return res

    note_names = hashed_frame.split(',')
    notes = [reverse_midi_notes[name] for name in note_names]
    for note_idx in notes:
        res[0, note_idx] = 1

    return res
