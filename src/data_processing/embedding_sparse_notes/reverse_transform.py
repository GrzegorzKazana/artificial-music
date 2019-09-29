import numpy as np
from scipy import sparse

from .common import unhash_named_frame, UNKNOWN_FRAME, TRACK_END
from ..common.helpers import unzip


def np2dicted(track, word_vectors):
    """
    Takes in track, and finds closest dicted vectors.
    2d np array (seq_length x embedding_dim) -> list of words, words similarities
    """
    words, similarities = unzip(
        [word_vectors.similar_by_vector(frame, topn=1)[0] for frame in track])

    return words, similarities


def handle_special_tokens(words):
    # TRACK_END handling - silencing frames after
    words_end_idx = words.index(
        TRACK_END) if TRACK_END in words else len(words)
    words_ended = words[:words_end_idx] + \
        ['' for _ in range(len(words - words_end_idx))]

    # UNKNOWN_FRAME handling - repeating previous frame
    words_known = [w if w != UNKNOWN_FRAME else (words_ended[i - 1] if i != 0 else '')
                   for i, w in enumerate(words_ended)]

    return words_known


def dicted2sparse(words):
    """
    takes in 1d np.array of note names (i.e. 'C5,A1'),
    returns 2d sparse array (seq_length x 128)
    """
    res = np.concatenate([unhash_named_frame(f) for f in words], axis=0)
    return sparse.coo_matrix(res)


def np2sparse(track, word_vectors):
    dicted, similarities = np2dicted(track, word_vectors)
    handled_dicted = handle_special_tokens(dicted)
    return dicted2sparse(handled_dicted), similarities
