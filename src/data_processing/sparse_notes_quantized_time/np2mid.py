import random
import numpy as np
from math import ceil
from mido import MidiFile, MidiTrack, Message, MetaMessage

from .config import MSECS_PER_FRAME, NUM_NOTES, NUM_VELOCITY
from ..common.helpers import flow, debug
from ..common.rw_np_mid import read_numpy_midi


def to_delta_time(messages):
    """
    takes messages with absolute timestamp, and returns list of 
    messages with delta time
    """
    messages_detlta_time = []
    now = 0
    for msg in messages:
        delta = int(msg.time - now)
        new_message = msg.copy(time=delta)
        messages_detlta_time.append(new_message)
        now = msg.time
    return messages_detlta_time


def total_decode(encoded_numpy):
    """
    takes in
    [[one_hot_encoded_note, velocities], ...] (n_of_frames x (one_hot_encoded_note + velocities))
    and transforms to
    [Messages with absolute time stamp]
    note_offs are encoded as note_ons with vel=0
    """
    messages = []
    # handle first frame
    notes_on = np.argwhere(encoded_numpy[0, :NUM_NOTES] == 1).flatten()
    for note in notes_on:
        vel = int(encoded_numpy[0, NUM_NOTES + note] *
                  128) if encoded_numpy.shape[1] > 128 else 127
        messages.append(Message('note_on', note=note,
                                velocity=vel, time=0))

    for i in range(1, len(encoded_numpy)):
        prev_frame = encoded_numpy[i-1]
        curr_frame = encoded_numpy[i]
        diff = curr_frame - prev_frame
        diff_notes = diff[:NUM_NOTES]
        notes_on = np.argwhere(diff_notes == 1).flatten()
        notes_off = np.argwhere(diff_notes == -1).flatten()

        for note in notes_on:
            time = i * MSECS_PER_FRAME
            vel = int(curr_frame[NUM_NOTES + note] *
                      128) if curr_frame.shape[0] > 128 else 127
            messages.append(Message(
                'note_on', note=note, velocity=vel, time=time))
        for note in notes_off:
            time = i * MSECS_PER_FRAME
            messages.append(Message('note_on', note=note,
                                    velocity=0, time=time))
    return messages


def messages_to_midifile(messages):
    """
    creates MidiFile with given messages
    """
    outfile = MidiFile()
    track = MidiTrack()
    outfile.tracks.append(track)
    for msg in messages:
        track.append(msg)
    return outfile


def np2mid(encoded_numpy):
    """
    takes in encoded numpy, returns MidiFile instance
    """
    return flow(
        total_decode,
        to_delta_time,
        messages_to_midifile
    )(encoded_numpy)
