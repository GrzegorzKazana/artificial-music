import numpy as np
import random

from .common import create_noise_adder

# seeds


def random_noise_seed(length, input_size, scaler=0.5, batch_size=16):
    return np.random.normal(size=(batch_size, length, input_size)) * scaler


def zero_seed(length, input_size, word_vectors, batch_size=16):
    zero_vector = word_vectors['']
    return np.tile(zero_vector, (batch_size, length, 1))


def const_frame_seed(length, input_size, word_vectors, batch_size=16):
    res = zero_seed(length, input_size, word_vectors, batch_size)
    for i in range(batch_size):
        note = word_vectors[random.choice(
            list(word_vectors.vocab.keys()))]
        res[i, :, :] = note

    return res


def short_frame_seed(length, input_size, word_vectors, batch_size=16, max_note_length=25):
    res = zero_seed(length, input_size, word_vectors, batch_size)
    for i in range(batch_size):
        note = word_vectors[random.choice(
            list(word_vectors.vocab.keys()))]
        note_on = np.random.randint(length)
        note_off = np.random.randint(
            note_on, min(note_on + max_note_length, length))
        res[i, note_on:note_off, :] = note

    return res


def multi_note_seed(length, input_size, word_vectors, batch_size=16, num_notes=5):
    res = zero_seed(length, input_size, word_vectors, batch_size)
    for j in range(batch_size):
        notes_keys = random.sample(list(word_vectors.vocab.keys()), num_notes)
        notes = [word_vectors[n] for n in notes_keys]
        note_change_times = np.sort(np.random.choice(
            np.arange(length), size=num_notes + 1))
        for i, note in enumerate(notes):
            res[j, note_change_times[i]:note_change_times[i + 1], :] = note

    return res


def multi_note_harmonic_seed(length, input_size, word_vectors, batch_size=16, num_notes=5):
    res = zero_seed(length, input_size, word_vectors, batch_size)
    base_note_key = random.choice(
        list(word_vectors.vocab.keys()))
    similar_notes_keys = [name for name, _ in word_vectors.similar_by_word(
        base_note_key, topn=num_notes - 1)]
    notes_keys = [base_note_key, *similar_notes_keys]
    notes = [word_vectors[k] for k in notes_keys]
    random.shuffle(notes)
    for j in range(batch_size):
        note_change_times = np.sort(np.random.choice(
            np.arange(length), size=num_notes + 1))
        for i, note in enumerate(notes):
            res[j, note_change_times[i]:note_change_times[i + 1], :] = note
    return res


random_noise_adder = create_noise_adder(random_noise_seed)

# refer to artifacts/embedded_seed_generators.json
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
