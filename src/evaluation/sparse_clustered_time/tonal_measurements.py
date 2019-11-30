import numpy as np


def tonal_range(sparse_track_notes):
    assert sparse_track_notes.ndim == 2
    pitches = sparse_track_notes.nonzero()[1]
    return pitches.max() - pitches.min()


def tonal_hist(sparse_track_notes):
    assert sparse_track_notes.ndim == 2
    return sparse_track_notes.sum(axis=0) / sparse_track_notes.sum()
