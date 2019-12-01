import numpy as np
from math import ceil
from mido import MidiFile, MidiTrack, Message
from sklearn.cluster import KMeans

from ..common.helpers import flow, debug
from ..sparse_notes_quantized_time.mid2np import to_absolute_time, filter_meta, note_off_to_zero_vel, to_raw_numpy, transform

COMMON_PPQ = 120
N_CLUSTERS = 24


def remove_subsequents_and_count_note_ticks(encoded_np):
    """
    removes subsequent notes, and counts subsequents lengths
    """
    idxs_of_change = np.insert(
        np.count_nonzero(np.diff(encoded_np, axis=0), axis=1).astype(np.bool), 0, True)
    notes = encoded_np[idxs_of_change]
    durations = np.diff(np.concatenate(
        (np.argwhere(idxs_of_change).reshape(-1), [encoded_np.shape[0]])))

    assert notes.shape[0] == durations.shape[0]
    return notes, durations


def ppq_to_quarters(durations):
    return np.around(durations / COMMON_PPQ, decimals=2)


def create_clustering_dict(db, durations):
    d = {}
    stats = []

    for l in np.unique(db.labels_):
        avg = durations[db.labels_ == l].mean()
        count = durations[db.labels_ == l].size
        min_ = durations[db.labels_ == l].min()
        max_ = durations[db.labels_ == l].max()
        std = durations[db.labels_ == l].std()

        stats.append({
            'avg': avg,
            'count': count,
            'min': min_,
            'max': max_,
            'std': std,
            'label': str(l),
        })

    sorted_stats = sorted(stats, key=lambda x: x['avg'])

    for i, s in enumerate(sorted_stats):
        d[str(i)] = s

    return d


def cluster_and_ohe_durations(durations):
    db = KMeans(n_clusters=N_CLUSTERS).fit(durations.reshape(-1, 1))
    durations_ohe = np.zeros((durations.size, N_CLUSTERS))

    durations_dict = create_clustering_dict(db, durations)
    db_label_to_dict_label = {
        v['label']: k for k, v in durations_dict.items()
    }

    for i, p in enumerate(db.predict(durations.reshape(-1, 1))):
        label_in_dict = db_label_to_dict_label[str(p)]
        durations_ohe[i, int(label_in_dict)] = 1

    return durations_ohe, db, durations_dict


def mid2np(mid, **kwargs):
    """
    takes in list of midi messages, returns encoded track in numpy format
    """
    track_ppq = mid.ticks_per_beat
    ppq_ratio = COMMON_PPQ / track_ppq

    # take messages from track, so timestamp
    # is not recalculated to seconds
    messages_of_valid_tracks = [
        [m for m in track]
        for track in mid.tracks if len(list(filter(lambda x: x.type in ['note_on', 'note_off'], track))) > 10
    ]

    def to_sparse(msgs):
        return flow(
            note_off_to_zero_vel,
            to_absolute_time,
            lambda msgs: [m.copy(time=int(m.time * ppq_ratio)) for m in msgs],
            filter_meta,
            to_raw_numpy,
            lambda x: transform(x, skip_velocity=True),
        )(msgs)

    def combine_sparses(sparses):
        largest = sparses[np.argmax([s.shape[0] for s in sparses])]
        res = np.zeros_like(largest)
        for sparse in sparses:
            # pylint: disable=unsupported-assignment-operation
            res[:sparse.shape[0], :] += sparse

        res = np.clip(res, 0, 1)
        return res

    return flow(
        lambda x: [to_sparse(t) for t in x],
        combine_sparses,
        remove_subsequents_and_count_note_ticks,
        lambda args: (args[0], ppq_to_quarters(args[1]))
    )(messages_of_valid_tracks)
