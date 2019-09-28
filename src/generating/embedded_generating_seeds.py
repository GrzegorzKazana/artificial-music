import numpy as np
import random

from .common import create_noise_adder

# seeds


def random_noise_seed(length, input_size, batch_size=16):
    return np.random.random((batch_size, length, input_size)) * 0.5


def zero_seed(length, input_size, batch_size=16):
    return np.zeros((batch_size, length, input_size))


def const_frame_seed(length, input_size, word_vectors, batch_size=16):
    res = np.zeros((batch_size, length, input_size))
    for i in range(batch_size):
        res[i, :, :] = random.choice(list(word_vectors.vocab.keys()))

    return res


def short_frame_seed(length, input_size, word_vectors, batch_size=16, max_note_length=25):
    res = np.zeros((batch_size, length, input_size))
    for i in range(batch_size):
        note_on = np.random.randint(length)
        note_off = np.random.randint(
            note_on, min(note_on + max_note_length, length))
        res[i, note_on:note_off, :] = random.choice(
            list(word_vectors.vocab.keys()))

    return res


def multi_note_seed(length, input_size, word_vectors, batch_size=16, num_notes=5):
    res = np.zeros((batch_size, length, input_size))
    for j in range(batch_size):
        notes = random.sample(list(word_vectors.vocab.keys()), num_notes)
        note_change_times = np.sort(np.random.choice(
            np.arange(length), size=num_notes + 1))
        for i, note in enumerate(notes):
            res[j, note_change_times[i]:note_change_times[i + 1], :] = note

    return res


def multi_note_harmonic_seed(length, input_size, word_vectors, batch_size=16, num_notes=5):
    res = np.zeros((batch_size, length, input_size))
    base_note = random.choice(list(word_vectors.vocab.keys()))
    similar_notes = [vec for name, vec in word_vectors.similar_by_word(
        base_note, topn=num_notes - 1)]
    notes = [base_note, *similar_notes]
    random.shuffle(notes)
    for j in range(batch_size):
        note_change_times = np.sort(np.random.choice(
            np.arange(length), size=num_notes + 1))
        for i, note in enumerate(notes):
            res[j, note_change_times[i]:note_change_times[i + 1], :] = note
    return res


random_noise_adder = create_noise_adder(random_noise_seed)

# refer to artifacts/
seed_generators = {
    "random_noise_seed": random_noise_seed,
    "zero_seed": zero_seed,
    "const_frame_seed": const_frame_seed,
    "short_frame_seed": short_frame_seed,
    "multi_note_seed": multi_note_seed,
    "multi_note_harmonic_seed": multi_note_harmonic_seed,
    "const_frame_seed_noise": random_noise_adder(const_frame_seed),
    "short_frame_seed_noise": random_noise_adder(short_frame_seed),
    "multi_note_seed_noise": random_noise_adder(multi_note_seed),
    "multi_note_harmonic_seed_noise": random_noise_adder(multi_note_harmonic_seed),
}
