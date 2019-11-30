import numpy as np
import zlib
from scipy import sparse

from src.data_processing.embedding_sparse_notes.common import hash_frame, map_hashed_frame_to_names


def rythm_range(duration_ohe):
    """
    expects ohe vectors, returns scalar
    """
    assert duration_ohe.ndim == 2
    durations = duration_ohe.nonzero()[1]
    return durations.max() - durations.min()


def rythm_hist(duration_ohe):
    """
    expects ohe vectors, returns histogram scaled to 1
    """
    assert duration_ohe.ndim == 2
    return duration_ohe.sum(axis=0) / duration_ohe.sum()


def rythm_compression(duration_ohe):
    """
    expects ohe vectors, return scalar
    """
    tokens = [str(s) for s in list(duration_ohe.nonzero()[1])]

    blob = ':'.join(tokens).encode()

    return len(blob) / len(zlib.compress(blob))


def rythm_transition_matrix(duration_ohe):
    """
    expects ohe vectors, return square matrix
    """
    idxs = list(duration_ohe.nonzero()[1])

    n = duration_ohe.shape[1]

    transition_matrix = np.zeros((n, n))

    for i, idx in enumerate(idxs[:-1]):
        idx_curr_token = idx
        idx_next_token = idxs[i + 1]
        transition_matrix[idx_curr_token, idx_next_token] += 1

    return transition_matrix / transition_matrix.max()
