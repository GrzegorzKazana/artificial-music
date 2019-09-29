import numpy as np
from scipy import sparse

from .common import unhash_named_frame
from ..common.helpers import unzip


def np2dicted(track, word_vectors):
    """
    Takes in track, and finds closest dicted vectors.
    2d np array (seq_length x embedding_dim) -> 1d np.array of words, words similarities
    """
    words, similarities = unzip(
        [word_vectors.similar_by_vector(frame, topn=1)[0] for frame in track])
    return np.array(words), np.array(similarities)


def dicted2sparse(track):
    """
    takes in 1d np.array of note names (i.e. 'C5,A1'),
    returns 2d sparse array (seq_length x 128)
    """
    res = np.concatenate([unhash_named_frame(f) for f in track], axis=0)
    return sparse.coo_matrix(res)


def np2sparse(track, word_vectors):
    dicted, similarities = np2dicted(track, word_vectors)
    return dicted2sparse(dicted), similarities
