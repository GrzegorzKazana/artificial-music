import numpy as np
import zlib
import itertools
from scipy import sparse

from src.data_processing.embedding_sparse_notes.common import hash_frame, map_hashed_frame_to_names


def tonal_range(sparse_track_notes):
    """
    expects m-out-out-n vectors, returns scalar
    """
    assert sparse_track_notes.ndim == 2
    pitches = sparse_track_notes.nonzero()[1]
    return pitches.max() - pitches.min()


def tonal_hist(sparse_track_notes):
    """
    expects m-out-out-n vectors, returns histogram scaled to 1
    """
    assert sparse_track_notes.ndim == 2

    padded_sparse = np.zeros((
        sparse_track_notes.shape[0],
        sparse_track_notes.shape[1] if sparse_track_notes.shape[1] % 12 == 0 else 12 * (
            sparse_track_notes.shape[1] // 12 + 1)
    ))
    padded_sparse[:, :sparse_track_notes.shape[1]] = sparse_track_notes

    n_splits = padded_sparse.shape[1] // 12
    stacked_by_octave = np.concatenate(
        np.split(padded_sparse, n_splits, axis=1), axis=0)

    return stacked_by_octave.sum(axis=0) / stacked_by_octave.sum()


def tonal_compression(sparse_track_notes):
    """
    expects m-out-out-n vectors, return scalar
    """
    tokens = [map_hashed_frame_to_names(hash_frame(sparse.csr_matrix(f)))
              for f in sparse_track_notes]

    blob = ':'.join(tokens).encode()

    return len(blob) / len(zlib.compress(blob))


def tonal_transition_matrix(sparse_track_notes):
    """
    expects m-out-out-n vectors, return square matrix
    """

    notes_in_each_step = np.split(
        np.argwhere(sparse_track_notes)[:, 1],
        np.cumsum(np.unique(np.argwhere(sparse_track_notes)[:, 0],
                            return_counts=True)[1])[:-1]
    )

    n = 12

    transition_matrix = np.zeros((n, n))

    for i, notes in enumerate(notes_in_each_step[:-1]):
        notes_curr = notes % 12
        notes_next = notes_in_each_step[i + 1] % 12
        for pair in itertools.product(notes_curr, notes_next):
            transition_matrix[pair[0], pair[1]] += 1

    return transition_matrix / transition_matrix.max()
