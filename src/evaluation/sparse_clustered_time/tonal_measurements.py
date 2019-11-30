import numpy as np
import zlib
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
    return sparse_track_notes.sum(axis=0) / sparse_track_notes.sum()


def tonal_compression(sparse_track_notes):
    """
    expects m-out-out-n vectors, return scalar
    """
    tokens = [map_hashed_frame_to_names(hash_frame(sparse.csr_matrix(f)))
              for f in sparse_track_notes]

    blob = ':'.join(tokens).encode()

    return len(blob) / len(zlib.compress(blob))


def tonal_transition_matrix(encoded_track_notes, wv):
    """
    expects embeddings of m-out-out-n vectors, return square matrix
    """
    tokens = [wv.similar_by_vector(f, topn=1)[0][0]
              for f in encoded_track_notes]

    n = len(wv.vocab.items())

    transition_matrix = np.zeros((n, n))

    for i, token in enumerate(tokens[:-1]):
        idx_curr_token = wv.vocab[token].index
        idx_next_token = wv.vocab[tokens[i + 1]].index
        transition_matrix[idx_curr_token, idx_next_token] += 1

    return transition_matrix / transition_matrix.max()
