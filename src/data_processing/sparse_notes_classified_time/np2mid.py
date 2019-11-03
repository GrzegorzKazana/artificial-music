import random
import numpy as np
from math import ceil
from mido import MidiFile, MidiTrack, Message, MetaMessage

from ..common.helpers import flow, debug
from src.data_processing.sparse_notes_quantized_time.np2mid import to_delta_time, total_decode
from src.data_processing.sparse_notes_classified_time.mid2np import COMMON_PPQ


def decode_durations(durations_ohe, clustering_dict):
    print(durations_ohe.shape)
    return np.array([clustering_dict[str(np.argmax(d))]['avg'] for d in durations_ohe])


def quarters_to_ppq(durations):
    return (durations * COMMON_PPQ).astype(np.int32)


def restore_subsequents_from_count(notes, durations):
    return np.concatenate([np.tile(n, (int(d), 1)) for n, d in zip(notes, durations)], axis=0)


def messages_to_midifile(messages, ticks_per_beat):
    """
    creates MidiFile with given messages
    """
    outfile = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    outfile.tracks.append(track)
    for msg in messages:
        track.append(msg)
    return outfile


def np2mid(encoded_numpy, durations_ohe, duration_dict, **kwargs):
    """
    takes in encoded numpy, returns MidiFile instance
    """

    return flow(
        lambda args: (args[0], decode_durations(args[1], duration_dict)),
        lambda args: (args[0], quarters_to_ppq(args[1])),
        lambda args: restore_subsequents_from_count(*args),
        lambda x: total_decode(x, 1),
        to_delta_time,
        lambda x: messages_to_midifile(x, COMMON_PPQ)
    )((encoded_numpy, durations_ohe))
