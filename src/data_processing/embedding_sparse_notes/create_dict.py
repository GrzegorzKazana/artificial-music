import numpy as np

from .common import UNKNOWN_FRAME, TRACK_END


def ignore_rarest_in_counter(counter, ignore_ratio):
    """
    transforms counter in such way, that skips keys which
    values are responsible for less than ignore_ratio of total
    assumes counter is ordered dict
    """
    total_occurences = sum(counter.values())
    accounted_occurences = total_occurences * (1 - ignore_ratio)

    accumulated_occurences = np.cumsum(np.array(list(counter.values())))
    max_accounted_index = np.max(np.argwhere(
        accumulated_occurences < accounted_occurences))

    return dict(list(counter.items())[:max_accounted_index])


def create_dict(counter, **kwargs):
    ignore_ratio = kwargs.get('ignore_ratio', 0.2)
    reduced_counter = ignore_rarest_in_counter(counter, ignore_ratio)

    # stats
    keys_total = len(counter.items())
    keys_accounted = len(reduced_counter.items())
    occurences_total = sum(counter.values())
    occurences_accounted = sum(reduced_counter.values())
    print(f'creating dict with {ignore_ratio} ignore_ratio: {keys_total} keys total ({occurences_total} occurences), {keys_accounted} keys accounted ({occurences_accounted} occurences)')

    return {
        UNKNOWN_FRAME: 0,
        TRACK_END: 1,
        **{note: i + 2 for i, note in enumerate(reduced_counter.keys())}
    }
