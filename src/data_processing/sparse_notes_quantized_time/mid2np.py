import numpy as np
from math import ceil
from mido import MidiFile, MidiTrack, Message

from .config import MSECS_PER_FRAME, NUM_NOTES, NUM_VELOCITY
from ..common.helpers import flow, debug
from ..common.rw_np_mid import save_numpy_midi


def to_absolute_time(messages):
    """
    takes in list of messages with delta-time timestamp,
    and returns list of messages with absolute timestamp
    """
    accumulated_time = 0
    messages_abs_time = []
    for msg in messages:
        accumulated_time += msg.time
        new_msg = msg.copy(time=accumulated_time)
        messages_abs_time.append(new_msg)
    return messages_abs_time


def filter_channel(channel_n, messages):
    """
    filters list of messages to only selected channel
    """
    return filter(lambda msg: msg.channel == channel_n, messages)


def filter_meta(messages):
    """
    filters list of messages to note_on and note_off events
    """
    return filter(lambda msg: msg.type == 'note_on' or msg.type == 'note_off', messages)


def note_off_to_zero_vel(messages):
    """
    changes note_off events to note_on events with 0 velocity
    """
    return map(lambda msg: msg.copy(type='note_on', velocity=0) if msg.type == 'note_off' else msg, messages)


def secs_to_msecs(messages):
    """
    transforms messages' timestamp from secs to msecs
    """
    return [msg.copy(time=1000 * msg.time) for msg in messages]


def encode_message(msg):
    """
    encodes mido message object to list
    """
    return [msg.note, msg.velocity, msg.time]


def to_raw_numpy(messages):
    """
    encodes all messages,
    and converts it to numpy array
    """
    return np.array([encode_message(msg) for msg in messages], dtype=np.float32)


def msecs_to_frames(raw_numpy):
    """
    processes time data from msecs to frame time,
    assumes time stamp is in last column
    """
    raw_numpy[:, -1] //= MSECS_PER_FRAME
    return raw_numpy


def snip_track(raw_numpy):
    """
    if first note starts not in first frame,
    every note is shifted
    """
    offset = raw_numpy[0, -1]
    raw_numpy[:, -1] -= offset
    return raw_numpy


def transform(raw_numpy):
    """
    transforms 
    [[note, velocity, time], ...]   (n_of_notes x 3)
    to
    [[one_hot_encoded_note, velocities], ...] (n_of_frames x (one_hot_encoded_note + velocities))
    """
    n_of_frames = int(raw_numpy[-1, -1])
    encoded = np.zeros((n_of_frames, NUM_NOTES + NUM_VELOCITY))
    for note, velocity, time in raw_numpy:
        note_int = int(note)
        time_int = int(time)
        if velocity == 0:
            encoded[time_int:, note_int] = 0
        else:
            encoded[time_int:, note_int] = 1
            encoded[time_int:, NUM_NOTES + note_int] = velocity / 128

    return encoded


def mid2np(messages):
    """
    takes in list of midi messages, returns encoded track in numpy format
    """
    return flow(
        note_off_to_zero_vel,
        secs_to_msecs,
        to_absolute_time,
        filter_meta,
        to_raw_numpy,
        msecs_to_frames,
        transform,
    )(messages)
