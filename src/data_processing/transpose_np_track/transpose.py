import numpy as np


def transpose(numpy_midi, steps):
    """
    shifts notes up/down by given step (1 = half-step in musical terms)
    may lose top #steps lowest/highiest notes 
    """
    # pylint: disable=unsupported-assignment-operation
    res = np.zeros_like(numpy_midi)
    fill_value = 0
    if steps > 0:
        res[:, :steps] = fill_value
        res[:, steps:] = numpy_midi[:, :-steps]
    elif steps < 0:
        res[:, steps:] = fill_value
        res[:, :steps] = numpy_midi[:, -steps:]
    else:
        res = numpy_midi
    return res
