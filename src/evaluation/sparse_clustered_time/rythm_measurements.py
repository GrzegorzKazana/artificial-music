import numpy as np
import zlib
from scipy import sparse

from src.data_processing.embedding_sparse_notes.common import hash_frame, map_hashed_frame_to_names


def rythm_range(sparse_track_notes):
    """
    expects ohe vectors, returns scalar
    """
    assert sparse_track_notes.ndim == 2
    durations = sparse_track_notes.nonzero()[1]
    return durations.max() - durations.min()


def rythm_hist(sparse_track_notes):
    """
    expects ohe vectors, returns histogram scaled to 1
    """
    assert sparse_track_notes.ndim == 2
    return sparse_track_notes.sum(axis=0) / sparse_track_notes.sum()


def rythm_compression(sparse_track_notes):
    """
    expects ohe vectors, return scalar
    """
    tokens = [str(s) for s in list(sparse_track_notes.nonzero()[1])]

    blob = ':'.join(tokens).encode()

    return len(blob) / len(zlib.compress(blob))
